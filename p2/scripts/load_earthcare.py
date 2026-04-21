import pathlib
import requests
from contextlib import contextmanager

import numpy as np
import xarray as xr
import fsspec
import stratify

from pystac_client import Client
from sklearn.neighbors import BallTree

io_params = {
    "fsspec_params": {
        "cache_type": "blockcache",
        "block_size": 8 * 1024 * 1024
    },
    "h5py_params": {
        "driver_kwds": {
            "rdcc_nbytes": 8 * 1024 * 1024
        }
    }
}

catalog_url = 'https://catalog.maap.eo.esa.int/catalogue/'
catalog = Client.open(catalog_url)
EC_COLLECTION = ['EarthCAREL2Validated_MAAP']

CREDENTIALS_FILE = (pathlib.Path.home() / "credentials.txt" ).resolve()   # Insert the .txt path

def load_credentials(file_path=CREDENTIALS_FILE):
    """Read key-value pairs from a credentials file into a dictionary."""
    creds = {}
    if not file_path.exists():
        raise FileNotFoundError(f"Credentials file not found: {file_path}")
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            creds[key.strip()] = value.strip()
    return creds


# --- ESA MAAP API ---

def get_token():
    """Use OFFLINE_TOKEN to fetch a short-lived access token."""
    creds = load_credentials()

    OFFLINE_TOKEN = creds.get("OFFLINE_TOKEN")
    CLIENT_ID = creds.get("CLIENT_ID")
    CLIENT_SECRET = creds.get("CLIENT_SECRET")
    # print(CLIENT_SECRET)

    if not all([OFFLINE_TOKEN, CLIENT_ID, CLIENT_SECRET]):
        raise ValueError("Missing OFFLINE_TOKEN, CLIENT_ID, or CLIENT_SECRET in credentials file")

    url = "https://iam.maap.eo.esa.int/realms/esa-maap/protocol/openid-connect/token"
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": OFFLINE_TOKEN,
        "scope": "offline_access openid"
    }

    response = requests.post(url, data=data)
    response.raise_for_status()

    response_json = response.json()
    access_token = response_json.get('access_token')

    if not access_token:
        raise RuntimeError("Failed to retrieve access token from IAM response")

    return access_token

token = get_token()
fs = fsspec.filesystem(
    "https", 
    headers={"Authorization": f"Bearer {token}"}, 
    **io_params["fsspec_params"], 
)

def search_ec_filename(product, orbit, frame):
    search = catalog.search(
        collections=EC_COLLECTION, 
        filter=f"(productType = '{product}') and orbitNumber = {orbit} and frame = '{frame}'", # For example filter by product type and orbitNumber. Use boolean logic for multi-filter queries
        method = 'GET', # This is necessary 
        max_items=1  # Adjust as needed, given the large amount of products it is recommended to set a limit if especially if you display results in pandas dataframe or similiar
    )
    items = list(search.items())
    if len(items):
        return items[0].assets.get('enclosure_h5').href

    raise ValueError(
        f'No EarthCARE files found for search {product=}, {orbit=}, {frame=}'
    )

from contextlib import contextmanager

@contextmanager
def read_ec_file(filename):
    try:
        f = fs.open(filename)
        ds = xr.open_dataset(
            f, 
            engine="h5netcdf", 
            **io_params["h5py_params"],  
            group="ScienceData"
        )
        yield ds
    finally:
        f.close()
        ds.close()

def select_vars(ds, data_vars):
    return ds.set_coords(["time", "latitude", "longitude", "height"])[data_vars]

def create_patch_ds(time, lat, lon, height):
    return xr.Dataset(
        coords = dict(
            time=xr.DataArray(time, dims="along_track"), 
            latitude=xr.DataArray(lat, dims="along_track"), 
            longitude=xr.DataArray(lon, dims="along_track"), 
            height=xr.DataArray(height, dims="height")
        ),
    )
    
def colocate_earthcare(lightning_patch, ds):
    patch_lat, patch_lon = lightning_patch.geometry.y, lightning_patch.geometry.x
    patch_ll = np.radians(np.stack(
        [patch_lat, patch_lon], axis=1
    ))
    
    ec_ll_tree = BallTree(
        np.radians(np.stack(
            [
                ds.latitude.values, 
                ds.longitude.values,
            ], axis=1
        )), 
        metric="haversine", 
    )

    _, neighbours = ec_ll_tree.query(patch_ll)

    return ds.isel(along_track=neighbours.ravel())
    
def regrid_height(ds, patch_ds):
    for var in ds.data_vars:
        da = ds[var]
        patch_ds[var] = (
            ("along_track", "height"), 
            stratify.interpolate(
                patch_ds.height.values[::-1],
                da.height.fillna(-np.inf).values,
                da.values,
                axis=1,
                rising=False
            )
        )
        patch_ds[var] = patch_ds[var].assign_attrs(ds[var].attrs)
    return patch_ds

def process_earthcare_patch(
    lightning_patch, cluster_id, product_vars: dict = None, save_path=pathlib.Path("./")
):
    orbitframe, instrument, cluster = cluster_id.split("_")
    orbit = orbitframe[:-1]
    frame = orbitframe[-1]

    lat, lon = lightning_patch.geometry.y, lightning_patch.geometry.x
    time = lightning_patch.time
    height = np.arange(50, 2e4, 100)
    patch_ds = create_patch_ds(time, lat, lon, height)

    patch_ds["lightning_count_2p5"] = (
        "along_track", lightning_patch.lightning_count_2p5
    )
    patch_ds["lightning_count_5"] = (
        "along_track", lightning_patch.lightning_count_5
    )
    
    for product, var in product_vars.items():
        with read_ec_file(search_ec_filename(product, orbit, frame)) as ds:
            ds = select_vars(ds, var)
            ds = colocate_earthcare(lightning_patch, ds)
            patch_ds = regrid_height(ds, patch_ds)

    save_name = f'earthcare_{cluster_id}.h5'
    patch_ds.to_netcdf(save_path/save_name, engine='h5netcdf')
    