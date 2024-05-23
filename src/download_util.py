import functools
import os
import shutil
from pathlib import Path
from typing import Optional, Callable, List, Iterable, Union, Tuple

import requests
from bs4 import BeautifulSoup
from joblib import delayed, Parallel
from requests import Response
from tqdm.auto import tqdm


def parse_download_request(
    uri: str, path: Path, filename: Optional[str] = None, verbose: bool = True, **kwargs
) -> Tuple[Optional[Response], Path, Path]:
    if filename is None:
        filename = uri.split('/')[-1]

    output_path = path / filename
    scrambled_output_path = path / f'~dl_{filename}.part'

    if os.path.exists(output_path):
        if verbose:
            print('- Already present')
        return None, output_path, scrambled_output_path

    r = requests.get(uri, stream=True, allow_redirects=True, **kwargs)
    # retry to authenticate on last redirect
    if r.status_code != 200:
        r = requests.get(r.url, stream=True, allow_redirects=True, **kwargs)

    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f'Request to {uri} returned status code {r.status_code}')

    return r, output_path, scrambled_output_path


def download_file_parallel(
    uri: str, path: Path, filename: Optional[str] = None, **kwargs
) -> Optional[Path]:
    r, output_path, scrambled_output_path = parse_download_request(
        uri, path, filename, verbose=False, **kwargs
    )
    if r is None:
        return output_path

    open(scrambled_output_path, 'wb').write(r.content)

    os.rename(scrambled_output_path, output_path)
    return output_path


def download_file(
    uri: str, path: Path, filename: Optional[str] = None, **kwargs
) -> Path:
    print(uri)

    r, output_path, scrambled_output_path = parse_download_request(
        uri, path, filename, verbose=True, **kwargs
    )
    if r is None:
        return output_path

    file_size = int(r.headers.get('Content-Length', 0))

    desc = '(Unknown total file size)' if file_size == 0 else ''
    r.raw.read = functools.partial(
        r.raw.read, decode_content=True
    )  # Decompress if needed
    with tqdm.wrapattr(r.raw, 'read', total=file_size, desc=desc) as r_raw:
        with scrambled_output_path.open('wb') as f:
            shutil.copyfileobj(r_raw, f)

    os.rename(scrambled_output_path, output_path)
    return output_path


def filter_empty(href: str) -> bool:
    return True


def get_hrefs(uri: str) -> List[str]:
    if not uri.endswith('/'):
        uri += '/'

    page = requests.get(uri)
    soup = BeautifulSoup(page.content, 'html.parser')

    hrefs = []
    for a in soup.find_all('a', href=True):
        href = a.attrs.get('href')
        if href.startswith('http') or href.startswith('ftp'):
            hrefs.append(href)
        else:
            hrefs.append(uri + href)
    return hrefs


def download_bulk(
    uris: Iterable[str],
    filenames: Union[str, Iterable[str]],
    paths: Union[Path, Iterable[Path]],
    n_jobs: int = 1,
    **kwargs,
) -> None:
    uris = list(uris)

    if isinstance(filenames, str):
        filenames = [filenames] * len(uris)
    if isinstance(paths, Path):
        paths = [paths] * len(uris)

    download_fun = download_file if n_jobs == 1 else download_file_parallel

    (
        Parallel(n_jobs=n_jobs, verbose=(n_jobs != 1) * 10)(
            delayed(download_fun)(uri=uri, path=path, filename=filename, **kwargs)
            for uri, path, filename in zip(uris, paths, filenames)
        )
    )


def download_all_links(
    uri: str,
    path: Path,
    filter_fun: Callable[[str], bool] = filter_empty,
    n_jobs: int = 1,
    **kwargs,
) -> None:
    uris = filter(filter_fun, get_hrefs(uri))
    download_bulk(uris, paths=path, n_jobs=n_jobs, **kwargs)
