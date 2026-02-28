from asyncio import sleep
from dataclasses import dataclass
from functools import cache
from time import perf_counter
from typing import TypedDict

import orjson
from aiohttp import ClientError, ClientSession
from loguru import logger
from yandex_music import DownloadInfo
from yandex_music.exceptions import NetworkError, NotFoundError, TimedOutError, UnauthorizedError, YandexMusicError

from nowplaying.core.config import config
from nowplaying.external.yandex import ClientAsync as YandexClientAsync
from nowplaying.util.dns import select_url
from nowplaying.util.http import STATUS_OK, get_headers
from nowplaying.util.url import urlparse


class SongQualityInfo(TypedDict):
    bit_depth: int | None
    bitrate_kbps: int
    sample_rate_khz: int
    highest_available: bool


@dataclass(frozen=True)
class DownloadedSong:
    file_extension: str
    thumbnail_url: str

    quality: SongQualityInfo
    duration_sec: int

    data: bytes

    platform_name: str = 'UNKNOWN'


class UdownloaderError(Exception):
    """Base class for all udownloader exceptions."""


class UdownloaderNetworkError(UdownloaderError):
    """Timeout, unreachable, etc."""


UNKNOWN_ERROR = f'Unknown error, contact @{config.DEVELOPER_USERNAME}'
YANDEX_SONGLINK_PATH_PREFIX = '/ya/'


@cache
def get_udownloader_base() -> str:
    return select_url(config.UDOWNLOADER_DOCKER_BASE_URL, config.UDOWNLOADER_BASE_URL)


def _extract_yandex_track_id(song_link_url: str) -> str | None:
    parsed = urlparse(song_link_url)
    path_parts = [part for part in parsed.path.split('/') if part]

    # song.link / album.link shorthand, for example: song.link/ya/123
    if parsed.netloc in {'song.link', 'album.link'} and parsed.path.startswith(YANDEX_SONGLINK_PATH_PREFIX):
        return path_parts[1] if len(path_parts) > 1 else None

    # direct Yandex Music URL
    if parsed.netloc.startswith('music.yandex.') and len(path_parts) > 1 and path_parts[-2] == 'track':
        return path_parts[-1]

    return None


def _song_quality_info_from_yandex(info: DownloadInfo, *, highest_available: bool) -> SongQualityInfo:
    return SongQualityInfo(
        bit_depth=16 if info.codec == 'flac' else None,
        bitrate_kbps=info.bitrate_in_kbps,
        # yandex api does not return sample rate in download_info
        sample_rate_khz=44,
        highest_available=highest_available,
    )


def _choose_download_info(download_infos: list[DownloadInfo], *, download_flac: bool) -> tuple[DownloadInfo, bool]:
    requested_codec = 'flac' if download_flac else 'mp3'
    filtered = [info for info in download_infos if info.codec == requested_codec]

    if not filtered and requested_codec == 'flac':
        filtered = [info for info in download_infos if info.codec == 'mp3']

    if not filtered:
        msg = 'No supported Yandex Music download formats are available'
        raise UdownloaderError(msg)

    best_for_codec = max(filtered, key=lambda item: item.bitrate_in_kbps)
    highest_global_bitrate = max(item.bitrate_in_kbps for item in download_infos)
    return best_for_codec, best_for_codec.bitrate_in_kbps == highest_global_bitrate


async def _download_from_yandex(song_link_url: str, *, yandex_token: str, download_flac: bool) -> DownloadedSong:
    track_id = _extract_yandex_track_id(song_link_url)
    if track_id is None:
        msg = 'Not a yandex track url'
        raise UdownloaderError(msg)

    start_time = perf_counter()
    client = YandexClientAsync(yandex_token)
    try:
        await client.init()
        tracks = await client.tracks(track_ids=[track_id])
        track = tracks[0] if tracks else None
        if track is None:
            msg = 'Track not found in Yandex Music'
            raise UdownloaderError(msg)

        download_infos = await track.get_download_info_async()
        if not download_infos:
            msg = 'Track is unavailable for downloading from Yandex Music'
            raise UdownloaderError(msg)

        selected_info, highest_available = _choose_download_info(
            download_infos,
            download_flac=download_flac,
        )
        data = await selected_info.download_bytes_async()

        thumbnail_url = ''
        if track.cover_uri:
            thumbnail_url = f'https://{track.cover_uri.replace("%%", "1000x1000")}'

        duration_sec = int((track.duration_ms or 0) / 1000)

        end = perf_counter()
        logger.info(
            f'Downloaded yandex track_id={track_id} in {(end - start_time) * 1000:.1f}ms '
            f'({selected_info.codec}/{selected_info.bitrate_in_kbps}kbps)'
        )
        return DownloadedSong(
            file_extension=selected_info.codec,
            thumbnail_url=thumbnail_url,
            quality=_song_quality_info_from_yandex(selected_info, highest_available=highest_available),
            duration_sec=duration_sec,
            data=data,
            platform_name='YANDEX_MUSIC',
        )
    except UnauthorizedError as err:
        msg = 'Yandex authorization expired, please re-login with /logout'
        raise UdownloaderError(msg) from err
    except (TimedOutError, NetworkError, ClientError, TimeoutError, OSError) as err:
        msg = 'yandex music is unavailable'
        raise UdownloaderNetworkError(msg) from err
    except NotFoundError as err:
        msg = 'Track not found in Yandex Music'
        raise UdownloaderError(msg) from err
    except YandexMusicError as err:
        msg = 'Failed to download track from Yandex Music'
        raise UdownloaderError(msg) from err


async def _download_by_songlink(body: dict[str, str | bool]) -> DownloadedSong:
    start_time = perf_counter()
    async with ClientSession(headers=get_headers()) as session:
        try:
            async with session.post(
                f'{get_udownloader_base()}/v1/download/by_songlink',
                json=body,
            ) as response:
                if response.status != STATUS_OK:
                    bytes_data = await response.read()
                    try:
                        json_data = orjson.loads(bytes_data)
                    except orjson.JSONDecodeError as err:
                        raise UdownloaderError(UNKNOWN_ERROR) from err

                    raise UdownloaderError(json_data.get('detail', UNKNOWN_ERROR))

                serve_time = response.headers['x-serve-time']
                thumbnail = response.headers['x-thumbnail-url']
                file_extension = response.headers['x-file-extension']
                duration_sec = int(response.headers['x-duration-seconds'])
                quality_json = orjson.loads(response.headers['x-file-quality'])
                platform_name = response.headers['x-downloaded-from']
                data = await response.read()
        except (ClientError, TimeoutError, OSError, orjson.JSONDecodeError) as err:
            err_msg = 'udownloader is unavailable'
            raise UdownloaderNetworkError(err_msg) from err

    end = perf_counter()
    logger.info(f'Downloaded {body} via udownloader in {(end - start_time) * 1000:.1f}ms ' f'(served in {serve_time})')
    return DownloadedSong(
        file_extension=file_extension,
        thumbnail_url=thumbnail,
        data=data,
        duration_sec=duration_sec,
        quality=quality_json,
        platform_name=platform_name,
    )


async def download(
    song_link_url: str, *, download_flac: bool, fast_route: bool, yandex_token: str | None = None
) -> DownloadedSong:
    if yandex_token and _extract_yandex_track_id(song_link_url):
        try:
            return await _download_from_yandex(
                song_link_url,
                yandex_token=yandex_token,
                download_flac=download_flac,
            )
        except UdownloaderError as err:
            logger.opt(exception=err).warning(
                f'Unable to download from yandex directly for {song_link_url!r}, fallback to udownloader'
            )

    last_exception: UdownloaderNetworkError | None = None

    for i in range(config.UDOWNLOADER_RETRIES):
        try:
            return await _download_by_songlink(
                {
                    'url': song_link_url,
                    'download_flac': download_flac,
                    'skip_song_link': fast_route,
                }
            )
        except UdownloaderNetworkError as err:
            # Retry only on network errors
            last_exception = err
            logger.opt(exception=err).warning(
                f'Got a network error while downloading {song_link_url!r}. Delaying for {i}s'
            )
            await sleep(i)
            continue

    if last_exception:
        raise last_exception

    raise UdownloaderNetworkError(UNKNOWN_ERROR)
