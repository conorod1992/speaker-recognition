# Speaker Recognition Add-on

Speaker recognition service for Home Assistant using voice embeddings.

This app (formerly called an add-on) supports 64-bit x86 (`amd64`) and 64-bit
Arm (`aarch64`) Home Assistant systems.

## Installation

1. In Home Assistant, open **Settings > Apps > Install app**.
2. Open the three-dot menu, choose **Repositories**, and add
   `https://github.com/conorod1992/speaker-recognition`.
3. Find **Speaker Recognition**, select **Install**, and wait for the local image
   build to finish.
4. Keep the default configuration for the first start, then select **Start**.
5. Open **Logs**. A healthy start reports the selected configuration, the
   Supervisor-discovered trusted local host addresses, loads the Resemblyzer
   voice encoder, and starts Uvicorn on `0.0.0.0:8099`.

The first installation can take several minutes because Supervisor builds the
CPU-only machine-learning runtime locally.

## About

This add-on provides a speaker recognition service that can identify speakers based on their voice characteristics. It uses voice embeddings to create unique speaker profiles and can identify speakers in real-time.

## Configuration

### Option: `host`

The network address to bind the service to.

- Default: `0.0.0.0`
- Type: `string`

Keep this at `0.0.0.0` so Home Assistant can reach the service.

### Option: `port`

The port the service listens on.

- Default: `8099`
- Type: `port`

The published container port is `8099`; keep this default unless you also know
how the integration will reach a non-default internal port.

### Option: `log_level`

The logging level for the service.

- Default: `info`
- Type: `list(debug|info|warning|error|critical)`

### Option: `access_log`

Enable or disable HTTP access logging.

- Default: `true`
- Type: `bool`

### Option: `embeddings_dir`

Directory where voice embeddings are stored.

- Default: `/share/speaker_recognition/embeddings`
- Type: `string`

### Option: `api_token`

Optional token for clients that are not running on the Home Assistant host.
The normal Home Assistant OS/Supervised integration path does not need a token:
the app asks Supervisor for the host's active interface addresses and trusts
only those same-host callers plus loopback/the Supervisor gateway. Arbitrary LAN
addresses are not trusted automatically.

Keep `allow_insecure_remote` disabled unless you intentionally want to expose
the API without authentication.

## API

The service exposes `/health`, `/train`, `/profiles/sync`, and `/recognize` on
the configured port. Requests originating from Home Assistant itself are
accepted without an API token. Requests from other LAN or remote devices require
the configured `api_token` and an `Authorization: Bearer ...` header (HTTP Basic
with the token as username or password is also supported).

The `/health` response includes `status`, `trained`, `enrolled_users`, and encoder
warm-up information. The profile fields let the integration restore recognition
availability without retraining on each Home Assistant restart.

## Data persistence

Speaker embeddings are stored in `/share/speaker_recognition/embeddings`.
Because `/share` is mapped read/write by Supervisor, embeddings survive app
container rebuilds and upgrades. Uninstalling the app does not remove files in
the shared folder automatically.

Multi-sample enrollment stores a normalized reference centroid per user together
with all accepted sample embeddings. Retraining one user rebuilds only that
profile. Legacy single-embedding `.npy` files are loaded for compatibility and
migrated when that user is retrained.

## Integration setup

Install the custom integration from this repository separately, restart Home
Assistant, then go to **Settings > Devices & services > Add integration** and
choose **Speaker Recognition**. Set the backend URL to
`http://HOME_ASSISTANT_IP:8099`; do not use `localhost`, because Home Assistant
Core and the app run in different containers.

For the normal Home Assistant OS/Supervised app path, leave **Backend API token**
blank. If you instead point the integration at a standalone backend on another
machine, configure `API_TOKEN` there and enter the same value in the integration.

Verify `/health` in the app logs before configuring voice samples.

## Development

Supervisor builds with `speaker_recognition_addon/` as the Docker context, so a
copy of the backend package is committed in this directory. The repository-root
package remains canonical. After changing it, run:

```shell
python scripts/sync_addon_sources.py
```

CI rejects a pull request if the committed copy drifts or if the Docker image
cannot be built from the actual Supervisor context.

## Support

For issues and feature requests, use
<https://github.com/conorod1992/speaker-recognition/issues>.
