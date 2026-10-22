# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Move media files between an agent server and the sandbox its harness runs in.

Harness CLIs read media from files, so an agent decodes prompt media with ``nemo_gym.core.split_text_and_media``,
writes it into the sandbox with ``upload_media`` and passes the paths to the harness.
"""

import posixpath
from collections.abc import Iterable

from nemo_gym.core.media import MediaFile, media_type_for_path
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.utils import download_bytes, upload_bytes


async def upload_media(sandbox: AsyncSandbox, files: Iterable[MediaFile], directory: str) -> list[str]:
    """Write each file into ``directory`` in the sandbox and return the sandbox paths, in order."""
    paths = []
    for index, file in enumerate(files):
        path = posixpath.join(directory, f"media-{index}.{file.extension}")
        await upload_bytes(sandbox, path=path, data=file.data)
        paths.append(path)
    return paths


async def download_media(sandbox: AsyncSandbox, path: str) -> MediaFile:
    """Read a media file from the sandbox, taking its media type from the file extension."""
    # Resolved before the transfer, so a file of an unsupported type is never downloaded.
    media_type = media_type_for_path(path)
    return MediaFile(media_type=media_type, data=await download_bytes(sandbox, path=path))
