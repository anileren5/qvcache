#!/bin/bash
set -euo pipefail

# This container is meant to be controlled externally by benchmark scripts.
# - PostgreSQL is started/stopped via docker exec + pg_ctl.
# - When using bind-mounted PGDATA on the host, file ownership must be consistent.
#   To avoid permission issues, we optionally create a runtime user that matches
#   the host UID/GID provided by docker run (-e HOST_UID / HOST_GID).

create_host_mapped_user() {
    local host_uid="${HOST_UID:-}"
    local host_gid="${HOST_GID:-}"
    local bench_user="${AK_BENCH_USER:-akerbench}"

    if [[ -z "${host_uid}" || -z "${host_gid}" ]]; then
        return 0
    fi

    # Create or reuse a group for HOST_GID.
    local group_name=""
    group_name="$(getent group "${host_gid}" | cut -d: -f1 || true)"
    if [[ -z "${group_name}" ]]; then
        groupadd -g "${host_gid}" "${bench_user}" >/dev/null 2>&1 || true
        group_name="${bench_user}"
    fi

    # Create or reuse a user for HOST_UID.
    if ! getent passwd "${host_uid}" >/dev/null 2>&1; then
        useradd -m -u "${host_uid}" -g "${group_name}" -s /bin/bash "${bench_user}" >/dev/null 2>&1 || true
    fi

    # Ensure the home directory exists.
    mkdir -p "/home/${bench_user}" >/dev/null 2>&1 || true
    chown -R "${host_uid}:${host_gid}" "/home/${bench_user}" >/dev/null 2>&1 || true
}

if [[ "$(id -u)" -eq 0 ]]; then
    create_host_mapped_user
fi

# Keep it alive unless an explicit command is provided.
if [[ $# -gt 0 ]]; then
    exec "$@"
fi

exec tail -f /dev/null
