#!/bin/bash

set -euo pipefail

#
# Helper utilities to reproduce legacy experiment safeguards:
# - restore a clean PGDATA snapshot (<datastore_clean> -> <datastore>) when configured
# - ALWAYS drop OS page cache for each benchmark run (legacy behavior)
# - start/stop PostgreSQL using pg_ctl when configured
# - export environment variables from INI so postgres can read them via getenv
# - collect new /tmp trace directories produced during a benchmark run
#
# Docker support (current refactor):
# - PostgreSQL server is executed inside a container.
# - Host-side scripts still manage the workload pipeline and storage directories.
# - To enable Docker mode you MUST specify a Docker image name.
#   (Non-Docker execution is treated as legacy and requires explicit opt-in.)
#

BENCH_ROOT="$(pwd)"
PROJECT_ROOT="$(cd "${BENCH_ROOT}/.." && pwd)"

# Docker runtime state (set by prepare_docker_environment()).
# NOTE: Do not reuse the same names as user-facing env vars.
AK_BENCH_DOCKER_IMAGE_INTERNAL=""
AK_BENCH_DOCKER_CONTAINER_INTERNAL=""
AK_BENCH_DOCKER_NETWORK_INTERNAL="host"
AK_BENCH_DOCKER_OS_USER_INTERNAL="akerbench"
AK_BENCH_DOCKER_NUMA_NODE_INTERNAL="0"
AK_BENCH_DOCKER_REMOVE_CONTAINER_ON_EXIT_INTERNAL="0"

declare -a AK_BENCH_DOCKER_ENV_ARGS=()

# Last pg_ctl stop stderr/stdout captured by pgctl_stop_graceful().
AK_BENCH_PGCTL_STOP_LAST_OUTPUT=""

resolve_config_path() {
    #
    # Resolve a config path.
    # - If the argument has no '/' and the file exists under configs/, use that.
    # - Otherwise, treat it as a relative/absolute path as provided.
    #
    local config_arg="$1"

    if [[ -z "${config_arg}" ]]; then
        printf ""
        return 0
    fi

    if [[ "${config_arg}" == /* ]]; then
        printf "%s" "${config_arg}"
        return 0
    fi

    if [[ "${config_arg}" != */* ]]; then
        local candidate="configs/${config_arg}"
        if [[ -f "${candidate}" ]]; then
            printf "%s" "${candidate}"
            return 0
        fi
    fi

    printf "%s" "${config_arg}"
}

resolve_path_under_root() {
    #
    # Resolve a path relative to the current working directory.
    #
    local path_arg="$1"

    if [[ -z "${path_arg}" ]]; then
        printf ""
        return 0
    fi

    if [[ "${path_arg}" == /* ]]; then
        printf "%s" "${path_arg}"
        return 0
    fi

    printf "%s" "${BENCH_ROOT}/${path_arg}"
}

resolve_datastore_path() {
    #
    # Resolve a postgres.datastore-like value into an absolute path.
    #
    # Convention:
    # - If value contains '/', it is treated as a relative path under BENCH_ROOT (unless absolute).
    # - Otherwise, it is treated as a datastore name under BENCH_ROOT/datastore/<name>.
    #
    local value="$1"

    if [[ -z "${value}" ]]; then
        printf ""
        return 0
    fi

    if [[ "${value}" == /* ]]; then
        printf "%s" "${value}"
        return 0
    fi

    if [[ "${value}" == */* ]]; then
        printf "%s" "${BENCH_ROOT}/${value}"
        return 0
    fi

    printf "%s" "${BENCH_ROOT}/datastore/${value}"
}

ini_get_value() {
    #
    # Read "key = value" from an INI file under a given section.
    # - Comments starting with '#' or ';' are ignored.
    # - Leading/trailing whitespace is trimmed.
    #
    # Args:
    #   $1: ini_path
    #   $2: section_name (without brackets)
    #   $3: key
    #
    local ini_path="$1"
    local section_name="$2"
    local key="$3"

    awk -v section="[${section_name}]" -v key="$key" '
        function ltrim(s) { sub(/^[ \t\r\n]+/, "", s); return s }
        function rtrim(s) { sub(/[ \t\r\n]+$/, "", s); return s }
        function trim(s)  { return rtrim(ltrim(s)) }

        BEGIN { in_section = 0 }

        {
            line = $0
            sub(/[;#].*$/, "", line)
            line = trim(line)
            if (line == "") { next }

            if (line ~ /^\[/) {
                in_section = (line == section)
                next
            }

            if (!in_section) { next }

            # Match: key = value (case-insensitive key comparison)
            eq = index(line, "=")
            if (eq == 0) { next }

            lhs = trim(substr(line, 1, eq - 1))
            if (tolower(lhs) != tolower(key)) { next }

            rhs = trim(substr(line, eq + 1))
            print rhs
            exit
        }
    ' "${ini_path}"
}

ini_list_section_kv() {
    #
    # Print key/value pairs from a section.
    # Output format: "key\tvalue" (tab separated)
    #
    # Args:
    #   $1: ini_path
    #   $2: section_name
    #
    local ini_path="$1"
    local section_name="$2"

    awk -v section="[${section_name}]" '
        function ltrim(s) { sub(/^[ \t\r\n]+/, "", s); return s }
        function rtrim(s) { sub(/[ \t\r\n]+$/, "", s); return s }
        function trim(s)  { return rtrim(ltrim(s)) }

        BEGIN { in_section = 0 }

        {
            line = $0
            sub(/[;#].*$/, "", line)
            line = trim(line)
            if (line == "") { next }

            if (line ~ /^\[/) {
                in_section = (line == section)
                next
            }

            if (!in_section) { next }

            # Match: key = value
            if (match(line, /^[A-Za-z_][A-Za-z0-9_]*[ \t]*=/)) {
                key = substr(line, 1, RLENGTH)
                sub(/[ \t]*=$/, "", key)
                sub(/^[A-Za-z_][A-Za-z0-9_]*[ \t]*=/, "", line)
                value = trim(line)
                print trim(key) "\t" value
            }
        }
    ' "${ini_path}"
}

log_info() {
    printf "[INFO] %s\n" "$*" >&2
}

log_warn() {
    printf "[WARN] %s\n" "$*" >&2
}

log_err() {
    printf "[ERROR] %s\n" "$*" >&2
}

require_cmd() {
    local cmd="$1"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        log_err "Required command not found in PATH: ${cmd}"
        exit 1
    fi
}

sanitize_container_component() {
    #
    # Convert an arbitrary string to a Docker container-name-safe token.
    #
    local raw="$1"
    printf "%s" "${raw}" \
        | tr '[:upper:]' '[:lower:]' \
        | sed -E 's/[^a-z0-9_.-]+/_/g'
}

docker_get_image() {
    local config_path="$1"

    if [[ -n "${AK_BENCH_DOCKER_IMAGE:-}" ]]; then
        printf "%s" "${AK_BENCH_DOCKER_IMAGE}"
        return 0
    fi

    local ini_image
    ini_image="$(ini_get_value "${config_path}" "docker" "image" || true)"
    printf "%s" "${ini_image}"
}

docker_is_legacy_opt_in() {
    [[ "${AK_BENCH_ALLOW_LEGACY_NO_DOCKER:-0}" == "1" ]]
}

docker_require_image_or_legacy() {
    local config_path="$1"

    local image
    image="$(docker_get_image "${config_path}")"

    if [[ -n "${image}" ]]; then
        return 0
    fi

    # Guard against the old pattern: docker.enabled=1 without specifying an image.
    local enabled
    enabled="$(ini_get_value "${config_path}" "docker" "enabled" || true)"
    if [[ "${enabled}" == "1" ]]; then
        log_err "docker.enabled=1 is deprecated. You must specify docker.image (or set AK_BENCH_DOCKER_IMAGE)."
        exit 1
    fi

    if docker_is_legacy_opt_in; then
        log_warn "Docker image not specified; running in legacy (non-Docker) mode due to AK_BENCH_ALLOW_LEGACY_NO_DOCKER=1"
        return 0
    fi

    log_err "Docker mode is required. Specify a Docker image name via:"
    log_err "  - environment: export AK_BENCH_DOCKER_IMAGE=<image:tag>"
    log_err "  - or config INI: [docker] image = <image:tag>"
    log_err "To run legacy host Postgres, set AK_BENCH_ALLOW_LEGACY_NO_DOCKER=1 (not recommended for review)."
    exit 1
}

docker_get_network() {
    local config_path="$1"

    if [[ -n "${AK_BENCH_DOCKER_NETWORK:-}" ]]; then
        printf "%s" "${AK_BENCH_DOCKER_NETWORK}"
        return 0
    fi

    local ini_network
    ini_network="$(ini_get_value "${config_path}" "docker" "network" || true)"
    if [[ -z "${ini_network}" ]]; then
        ini_network="host"
    fi
    printf "%s" "${ini_network}"
}

docker_get_numa_node() {
    local config_path="$1"

    if [[ -n "${AK_BENCH_DOCKER_NUMA_NODE:-}" ]]; then
        printf "%s" "${AK_BENCH_DOCKER_NUMA_NODE}"
        return 0
    fi

    local ini_node
    ini_node="$(ini_get_value "${config_path}" "docker" "numa_node" || true)"
    if [[ -z "${ini_node}" ]]; then
        ini_node="0"
    fi

    printf "%s" "${ini_node}"
}

docker_get_remove_container_on_exit() {
    local config_path="$1"

    if [[ -n "${AK_BENCH_DOCKER_REMOVE_CONTAINER_ON_EXIT:-}" ]]; then
        printf "%s" "${AK_BENCH_DOCKER_REMOVE_CONTAINER_ON_EXIT}"
        return 0
    fi

    local ini_value
    ini_value="$(ini_get_value "${config_path}" "docker" "remove_container_on_exit" || true)"
    if [[ -n "${ini_value}" ]]; then
        printf "%s" "${ini_value}"
        return 0
    fi

    # Legacy alias: rm_on_exit means removing the *container*, not the image.
    ini_value="$(ini_get_value "${config_path}" "docker" "rm_on_exit" || true)"
    if [[ -n "${ini_value}" ]]; then
        printf "%s" "${ini_value}"
        return 0
    fi

    printf "0"
}

docker_get_container_name() {
    local config_path="$1"
    local image_tag="$2"
    local host_port="$3"

    if [[ -n "${AK_BENCH_DOCKER_CONTAINER_NAME:-}" ]]; then
        printf "%s" "${AK_BENCH_DOCKER_CONTAINER_NAME}"
        return 0
    fi

    local ini_name
    ini_name="$(ini_get_value "${config_path}" "docker" "container_name" || true)"
    if [[ -n "${ini_name}" ]]; then
        printf "%s" "${ini_name}"
        return 0
    fi

    local token
    token="$(sanitize_container_component "${image_tag}")"
    printf "akerbench_%s_%s" "${token}" "${host_port}"
}

docker_is_enabled_for_config() {
    local config_path="$1"
    local image
    image="$(docker_get_image "${config_path}")"
    [[ -n "${image}" ]]
}

docker_ensure_container_running() {
    local config_path="$1"

    if ! docker_is_enabled_for_config "${config_path}"; then
        return 0
    fi

    require_cmd docker

    local image_tag
    image_tag="$(docker_get_image "${config_path}")"

    local host_port
    host_port="$(ini_get_value "${config_path}" "postgres" "port" || true)"
    if [[ -z "${host_port}" ]]; then
        host_port="5432"
    fi

    local container_name
    container_name="$(docker_get_container_name "${config_path}" "${image_tag}" "${host_port}")"

    local network_mode
    network_mode="$(docker_get_network "${config_path}")"

    local run_script
    run_script="${PROJECT_ROOT}/docker/scripts/run_container.sh"
    if [[ ! -x "${run_script}" ]]; then
        log_err "Docker run script not found or not executable: ${run_script}"
        exit 1
    fi

    "${run_script}" "${image_tag}" "${container_name}" "${network_mode}" "${host_port}"

    AK_BENCH_DOCKER_IMAGE_INTERNAL="${image_tag}"
    AK_BENCH_DOCKER_CONTAINER_INTERNAL="${container_name}"
    AK_BENCH_DOCKER_NETWORK_INTERNAL="${network_mode}"
    AK_BENCH_DOCKER_OS_USER_INTERNAL="${AK_BENCH_DOCKER_OS_USER:-akerbench}"
    AK_BENCH_DOCKER_NUMA_NODE_INTERNAL="$(docker_get_numa_node "${config_path}")"
    AK_BENCH_DOCKER_REMOVE_CONTAINER_ON_EXIT_INTERNAL="$(docker_get_remove_container_on_exit "${config_path}")"

    docker_resolve_exec_user
}

prepare_docker_environment() {
    #
    # Initialize Docker runtime state for a given benchmark config.
    # If Docker is required (default), this validates docker.image is set.
    #
    local config_path="$1"

    docker_require_image_or_legacy "${config_path}"

    if docker_is_enabled_for_config "${config_path}"; then
        docker_ensure_container_running "${config_path}"
    fi
}

docker_exec_root() {
    #
    # Execute a shell command inside the benchmark container as root.
    # This is used for user/group discovery and recovery when the configured
    # non-root execution user does not exist.
    #
    local inner_cmd="$1"

    docker exec -u 0 \
        "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" \
        bash -lc "cd '${BENCH_ROOT}' && ${inner_cmd}"
}

docker_resolve_exec_user() {
    #
    # Resolve a usable execution user inside the container without rebuilding
    # images.
    #
    # The refactored harness assumes a stable username (default: akerbench), but
    # the container entrypoint may decide to reuse an existing UID mapping and
    # skip creating that exact username. This resolver:
    #   1) Uses the preferred username if it exists.
    #   2) Falls back to any existing username mapped to the host UID.
    #   3) Attempts to create the preferred username with host UID/GID.
    #   4) Falls back to numeric uid:gid as a last resort.
    #
    if [[ -z "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
        return 0
    fi

    local preferred_user="${AK_BENCH_DOCKER_OS_USER_INTERNAL}"
    if [[ "${preferred_user}" =~ ^[0-9]+(:[0-9]+)?$ ]]; then
        log_info "Docker exec user specified as numeric uid[:gid]: ${preferred_user}"
        return 0
    fi

    if docker_exec_root "getent passwd '${preferred_user}' >/dev/null 2>&1"; then
        return 0
    fi

    local host_uid
    local host_gid
    host_uid="$(id -u)"
    host_gid="$(id -g)"

    log_warn "Docker user '${preferred_user}' not found in container; resolving (HOST_UID=${host_uid}, HOST_GID=${host_gid})"

    local uid_line=""
    uid_line="$(docker_exec_root "getent passwd '${host_uid}' 2>/dev/null" || true)"
    if [[ -n "${uid_line}" ]]; then
        local existing_user
        existing_user="${uid_line%%:*}"
        AK_BENCH_DOCKER_OS_USER_INTERNAL="${existing_user}"
        log_warn "Using existing container user '${existing_user}' for HOST_UID=${host_uid}"
        return 0
    fi

    if ! docker_exec_root "command -v useradd >/dev/null 2>&1 && command -v groupadd >/dev/null 2>&1"; then
        AK_BENCH_DOCKER_OS_USER_INTERNAL="${host_uid}:${host_gid}"
        log_warn "Container lacks useradd/groupadd; falling back to numeric uid:gid '${AK_BENCH_DOCKER_OS_USER_INTERNAL}'"
        log_warn "Some tools may require a passwd entry; consider fixing the image entrypoint if this fails"
        return 0
    fi

    local group_line=""
    local group_name=""
    group_line="$(docker_exec_root "getent group '${host_gid}' 2>/dev/null" || true)"
    if [[ -n "${group_line}" ]]; then
        group_name="${group_line%%:*}"
    else
        group_name="${preferred_user}"

        # Try to create the group with the host GID. If the name conflicts,
        # retry with a deterministic alternative.
        docker_exec_root "groupadd -g '${host_gid}' '${group_name}' >/dev/null 2>&1" || true
        group_line="$(docker_exec_root "getent group '${host_gid}' 2>/dev/null" || true)"
        if [[ -n "${group_line}" ]]; then
            group_name="${group_line%%:*}"
        else
            group_name="${preferred_user}_${host_gid}"
            docker_exec_root "groupadd -g '${host_gid}' '${group_name}' >/dev/null 2>&1" || true
            group_line="$(docker_exec_root "getent group '${host_gid}' 2>/dev/null" || true)"
            if [[ -n "${group_line}" ]]; then
                group_name="${group_line%%:*}"
            else
                # As a last attempt, create the group without forcing the GID.
                docker_exec_root "groupadd '${group_name}' >/dev/null 2>&1" || true
            fi
        fi
    fi

    docker_exec_root "useradd -m -u '${host_uid}' -g '${group_name}' -s /bin/bash '${preferred_user}' >/dev/null 2>&1" || true

    if docker_exec_root "getent passwd '${preferred_user}' >/dev/null 2>&1"; then
        AK_BENCH_DOCKER_OS_USER_INTERNAL="${preferred_user}"
        log_info "Created container user '${preferred_user}' (uid=${host_uid}, gid=${host_gid}) for benchmark execution"
        return 0
    fi

    AK_BENCH_DOCKER_OS_USER_INTERNAL="${host_uid}:${host_gid}"
    log_warn "Failed to create or resolve a named container user; falling back to numeric uid:gid '${AK_BENCH_DOCKER_OS_USER_INTERNAL}'"
    log_warn "Some tools may require a passwd entry; consider fixing the image entrypoint if this fails"
}

docker_exec_raw() {
    #
    # Execute a shell command inside the benchmark container.
    #
    local inner_cmd="$1"

    docker exec -u "${AK_BENCH_DOCKER_OS_USER_INTERNAL}" \
        "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" \
        bash -lc "cd '${BENCH_ROOT}' && ${inner_cmd}"
}

docker_exec_raw_with_env() {
    #
    # Execute a shell command inside the benchmark container with environment variables
    # propagated from [env] section.
    #
    local inner_cmd="$1"

    docker exec -u "${AK_BENCH_DOCKER_OS_USER_INTERNAL}" \
        "${AK_BENCH_DOCKER_ENV_ARGS[@]}" \
        "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" \
        bash -lc "cd '${BENCH_ROOT}' && ${inner_cmd}"
}

export_env_from_ini() {
    #
    # Export environment variables under [env] section so PostgreSQL can read them via getenv.
    # Keys must be valid shell variable names.
    #
    # Args:
    #   $1: config_path
    #   $2: optional AKER_CONFIG_PATH override (wins over INI)
    #
    local config_path="$1"
    local aker_config_override="${2:-}"

    AK_BENCH_DOCKER_ENV_ARGS=()

    while IFS=$'\t' read -r key value; do
        if [[ -z "${key}" ]]; then
            continue
        fi
        export "${key}=${value}"
        log_info "Exported env: ${key}=${value}"

        if [[ -n "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
            AK_BENCH_DOCKER_ENV_ARGS+=( -e "${key}=${value}" )
        fi
    done < <(ini_list_section_kv "${config_path}" "env" || true)

    if [[ -n "${aker_config_override}" ]]; then
        local resolved
        resolved="$(resolve_path_under_root "${aker_config_override}")"
        export AKER_CONFIG_PATH="${resolved}"
        log_info "Exported env override: AKER_CONFIG_PATH=${resolved}"

        if [[ -n "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
            AK_BENCH_DOCKER_ENV_ARGS+=( -e "AKER_CONFIG_PATH=${resolved}" )
        fi
    fi
}

pg_initdb() {
    #
    # Initialize a new PGDATA directory.
    #
    # Args:
    #   $1: datastore_path
    #   $2: db_superuser
    #
    local datastore_path="$1"
    local db_superuser="$2"

    if [[ -z "${db_superuser}" ]]; then
        log_warn "postgres.user is empty; initdb will use the OS user as the database superuser"
    fi

    if [[ -n "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
        if [[ -n "${db_superuser}" ]]; then
            docker_exec_raw "initdb -D '${datastore_path}' -U '${db_superuser}'"
        else
            docker_exec_raw "initdb -D '${datastore_path}'"
        fi
        return 0
    fi

    require_cmd initdb
    if [[ -n "${db_superuser}" ]]; then
        initdb -D "${datastore_path}" -U "${db_superuser}"
    else
        initdb -D "${datastore_path}"
    fi
}

pgctl_stop_graceful() {
    #
    # Gracefully stop PostgreSQL using pg_ctl for a specific PGDATA.
    #
    # Policy:
    # - This function does NOT exit the script.
    # - It returns non-zero only when pg_ctl fails for reasons other than
    #   "server not running".
    # - "server not running" cases are treated as success to keep stop idempotent.
    #
    local datastore_path="$1"

    AK_BENCH_PGCTL_STOP_LAST_OUTPUT=""

    if [[ -z "${datastore_path}" ]]; then
        log_warn "datastore path is empty; skip pg_ctl stop"
        return 0
    fi

    if [[ ! -d "${datastore_path}" ]]; then
        log_warn "datastore path not found; skip pg_ctl stop: ${datastore_path}"
        return 0
    fi

    local stop_output=""
    if [[ -n "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
        if stop_output="$(docker_exec_raw "pg_ctl -D '${datastore_path}' -m fast stop 2>&1")"; then
            return 0
        fi
    else
        if ! command -v pg_ctl >/dev/null 2>&1; then
            AK_BENCH_PGCTL_STOP_LAST_OUTPUT="pg_ctl not found"
            return 1
        fi

        if stop_output="$(pg_ctl -D "${datastore_path}" -m fast stop 2>&1)"; then
            return 0
        fi
    fi

    # Treat "not running"-style stop failures as success.
    if [[ "${stop_output}" == *"no server running"* ]]; then
        log_warn "pg_ctl reports no server running; treat stop as success: ${datastore_path}"
        return 0
    fi

    if [[ "${stop_output}" == *"PID file"* && "${stop_output}" == *"does not exist"* ]]; then
        log_warn "pg_ctl reports missing PID file; treat stop as success: ${datastore_path}"
        return 0
    fi

    AK_BENCH_PGCTL_STOP_LAST_OUTPUT="${stop_output}"
    return 1
}

pgctl_stop_force_all() {
    #
    # Force-stop PostgreSQL processes for "don't care" paths.
    #
    # This is intended for snapshot recovery / rm+cp operations where:
    # - the target PGDATA may not be the one currently running, and
    # - postgres may not be running at all.
    #
    # Policy:
    # - MUST NOT exit the script.
    # - MUST return success (0) even if nothing is running.
    # - In Docker mode, this stops *all* postgres processes in the container.
    #
    local datastore_path="$1"

    if [[ -n "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
        # Stop a specific PGDATA if it exists, then kill any remaining postgres processes.
        docker_exec_root "\
            set +e; \
            if [[ -n '${datastore_path}' && -d '${datastore_path}' ]]; then \
                pg_ctl -D '${datastore_path}' -m immediate stop >/dev/null 2>&1 || true; \
            fi; \
            pids=\$(ps -eo pid=,comm= 2>/dev/null | awk '\$2==\"postgres\" {print \$1}'); \
            if [[ -n \"\${pids}\" ]]; then \
                kill -TERM \${pids} >/dev/null 2>&1 || true; \
                sleep 2; \
                kill -KILL \${pids} >/dev/null 2>&1 || true; \
            fi; \
            true\
        " >/dev/null 2>&1 || true
        return 0
    fi

    # Non-Docker mode: do not kill all host postgres processes.
    if [[ -n "${datastore_path}" && -d "${datastore_path}" ]] && command -v pg_ctl >/dev/null 2>&1; then
        pg_ctl -D "${datastore_path}" -m immediate stop >/dev/null 2>&1 || true
    fi

    # If a postmaster.pid exists, try to kill that PID as a last resort.
    if [[ -n "${datastore_path}" && -f "${datastore_path}/postmaster.pid" ]]; then
        local pid=""
        pid="$(head -n 1 "${datastore_path}/postmaster.pid" 2>/dev/null || true)"
        if [[ -n "${pid}" ]]; then
            kill -TERM "${pid}" >/dev/null 2>&1 || true
            sleep 1
            kill -KILL "${pid}" >/dev/null 2>&1 || true
        fi
    fi

    return 0
}

pgctl_stop() {
    local datastore_path="$1"

    if pgctl_stop_graceful "${datastore_path}"; then
        return 0
    fi

    local stop_output="${AK_BENCH_PGCTL_STOP_LAST_OUTPUT}"

    if [[ -n "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
        log_err "pg_ctl stop failed (docker mode) for PGDATA=${datastore_path}: ${stop_output}"
        log_err "Aborting benchmark due to unsafe PGDATA operations"

        # Stop the container to avoid leaving stray processes, but keep it for log collection.
        if [[ -n "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
            log_warn "Stopping benchmark container due to pg_ctl stop failure (keeping container for log collection): ${AK_BENCH_DOCKER_CONTAINER_INTERNAL}"
            docker stop "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" >/dev/null 2>&1 || true
        fi

        exit 1
    fi

    log_err "pg_ctl stop failed for PGDATA=${datastore_path}: ${stop_output}"
    log_err "Aborting benchmark due to unsafe PGDATA operations"
    exit 1
}

pgctl_start() {
    local datastore_path="$1"
    local psql_config_path="$2"
    local pg_log_path="$3"

    if [[ ! -d "${datastore_path}" ]]; then
        log_err "datastore path not found; cannot pg_ctl start: ${datastore_path}"
        exit 1
    fi

    mkdir -p "$(dirname -- "${pg_log_path}")"

    if [[ -n "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
        local numa_node="${AK_BENCH_DOCKER_NUMA_NODE_INTERNAL}"

        local extra_opts=""
        if [[ -n "${psql_config_path}" ]]; then
            extra_opts="${extra_opts} -o --config-file=${psql_config_path}"
        fi

        docker_exec_raw_with_env "\
            set -euo pipefail; \
            numa_prefix=(); \
            if [[ -n '${numa_node}' ]]; then numa_prefix=(numactl --cpunodebind='${numa_node}' --membind='${numa_node}'); fi; \
            \"\${numa_prefix[@]}\" pg_ctl -D '${datastore_path}' -l '${pg_log_path}' ${extra_opts} start >/dev/null\
        "
        return 0
    fi

    require_cmd pg_ctl

    if [[ -n "${psql_config_path}" ]]; then
        pg_ctl -D "${datastore_path}" -l "${pg_log_path}" -o "--config-file=${psql_config_path}" start >/dev/null
    else
        pg_ctl -D "${datastore_path}" -l "${pg_log_path}" start >/dev/null
    fi
}

pg_wait_ready() {
    local host="$1"
    local port="$2"
    local user="$3"

    if [[ -z "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
        require_cmd pg_isready
    fi

    if [[ -z "${user}" ]]; then
        log_warn "postgres.user is empty; pg_isready will use the OS user by default"
    fi

    local max_tries=60
    local i=0
    while [[ ${i} -lt ${max_tries} ]]; do
        if [[ -n "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
            if [[ -n "${user}" ]]; then
                if docker_exec_raw "pg_isready -h '${host}' -p '${port}' -U '${user}'" >/dev/null 2>&1; then
                    return 0
                fi
            else
                if docker_exec_raw "pg_isready -h '${host}' -p '${port}'" >/dev/null 2>&1; then
                    return 0
                fi
            fi
        else
            if [[ -n "${user}" ]]; then
                if pg_isready -h "${host}" -p "${port}" -U "${user}" >/dev/null 2>&1; then
                    return 0
                fi
            else
                if pg_isready -h "${host}" -p "${port}" >/dev/null 2>&1; then
                    return 0
                fi
            fi
        fi
        sleep 1
        i=$((i + 1))
    done

    log_err "PostgreSQL did not become ready (host=${host}, port=${port})"
    return 1
}

drop_os_page_cache() {
    #
    # Legacy requirement: always drop OS page cache for each benchmark run.
    # Prompts for sudo password are allowed.
    #
    log_info "Dropping OS page cache"

    if [[ "$(id -u)" -eq 0 ]]; then
        sync
        echo 3 > /proc/sys/vm/drop_caches
        return 0
    fi

    require_cmd sudo

    # This may prompt for a password; that is intended.
    sudo sync
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
}

ensure_clean_snapshot() {
    #
    # Create <datastore_clean_path> if missing by copying <datastore_path>.
    # This can be expensive on large datasets.
    #
    # Args:
    #   $1: config_path (for exporting [env] before restarting postgres)
    #   $2: datastore_path
    #   $3: datastore_clean_path
    #   $4: psql_config_path
    #   $5: pg_log_path
    #   $6: host
    #   $7: port
    #   $8: user
    #
    local config_path="$1"
    local datastore_path="$2"
    local datastore_clean_path="$3"
    local psql_config_path="$4"
    local pg_log_path="$5"
    local host="$6"
    local port="$7"
    local user="$8"
    local aker_config_override="${9:-}"

    if [[ -d "${datastore_clean_path}" ]]; then
        return 0
    fi

    if [[ "${AK_BENCH_CREATE_CLEAN_SNAPSHOT_IF_MISSING:-0}" != "1" ]]; then
        log_err "Clean snapshot not found: ${datastore_clean_path}"
        log_err "Set AK_BENCH_CREATE_CLEAN_SNAPSHOT_IF_MISSING=1 to create it automatically."
        exit 1
    fi

    log_info "Creating clean snapshot: ${datastore_clean_path}"

    pgctl_stop_force_all "${datastore_path}"

    if [[ ! -d "${datastore_path}" ]]; then
        log_err "Cannot create clean snapshot; datastore does not exist: ${datastore_path}"
        exit 1
    fi

    rm -rf "${datastore_clean_path}"
    cp -r "${datastore_path}" "${datastore_clean_path}"

    # Bring the server back up for subsequent steps that query PostgreSQL.
    export_env_from_ini "${config_path}" "${aker_config_override}"

    pgctl_start "${datastore_path}" "${psql_config_path}" "${pg_log_path}"
    pg_wait_ready "${host}" "${port}" "${user}"
}

restore_clean_snapshot_and_start() {
    #
    # If postgres.datastore is set:
    # - ensure a clean snapshot exists
    # - restore <datastore_clean> -> <datastore>
    # - ALWAYS drop OS caches
    # - export env vars from [env]
    # - start PostgreSQL from the restored data directory
    #
    local config_path="$1"
    local purpose="$2"  # for log text only
    local aker_config_override="${3:-}"

    prepare_docker_environment "${config_path}"

    local datastore_path
    datastore_path="$(ini_get_value "${config_path}" "postgres" "datastore")"
    datastore_path="$(resolve_datastore_path "${datastore_path}")"
    if [[ -z "${datastore_path}" ]]; then
        # No PGDATA configured; caller likely manages Postgres externally.
        return 0
    fi

    local datastore_clean_path
    datastore_clean_path="$(ini_get_value "${config_path}" "postgres" "datastore_clean")"
    if [[ -z "${datastore_clean_path}" ]]; then
        datastore_clean_path="${datastore_path}-clean"
    fi

    datastore_clean_path="$(resolve_datastore_path "${datastore_clean_path}")"

    local psql_config_path
    psql_config_path="$(ini_get_value "${config_path}" "postgres" "psql_config")"

    if [[ -z "${psql_config_path}" && -f "${BENCH_ROOT}/configs/postgresql.conf" ]]; then
        psql_config_path="configs/postgresql.conf"
    fi

    psql_config_path="$(resolve_path_under_root "${psql_config_path}")"

    local pg_log_path
    pg_log_path="$(ini_get_value "${config_path}" "postgres" "pg_log")"
    if [[ -z "${pg_log_path}" ]]; then
        pg_log_path="output/postgres.log"
    fi
    pg_log_path="$(resolve_path_under_root "${pg_log_path}")"

    local host
    host="$(ini_get_value "${config_path}" "postgres" "host")"
    local port
    port="$(ini_get_value "${config_path}" "postgres" "port")"
    local user
    user="$(ini_get_value "${config_path}" "postgres" "user")"

    # Ensure a clean snapshot exists (optionally create it).
    ensure_clean_snapshot \
        "${config_path}" \
        "${datastore_path}" \
        "${datastore_clean_path}" \
        "${psql_config_path}" \
        "${pg_log_path}" \
        "${host}" \
        "${port}" \
        "${user}" \
        "${aker_config_override}"

    log_info "Restoring clean snapshot for ${purpose}: ${datastore_clean_path} -> ${datastore_path}"

    pgctl_stop_force_all "${datastore_path}"

    rm -rf "${datastore_path}"
    cp -r "${datastore_clean_path}" "${datastore_path}"

    drop_os_page_cache

    export_env_from_ini "${config_path}" "${aker_config_override}"

    pgctl_start "${datastore_path}" "${psql_config_path}" "${pg_log_path}"
    pg_wait_ready "${host}" "${port}" "${user}"
}

restore_clean_snapshot_and_start_no_cache_drop() {
    #
    # Same as restore_clean_snapshot_and_start(), but does NOT drop OS page cache.
    # This is useful for workload generation or index build steps where OS cache state
    # is not part of the measured benchmark.
    #
    local config_path="$1"
    local purpose="$2"  # for log text only
    local aker_config_override="${3:-}"

    prepare_docker_environment "${config_path}"

    local datastore_path
    datastore_path="$(ini_get_value "${config_path}" "postgres" "datastore")"
    datastore_path="$(resolve_datastore_path "${datastore_path}")"
    if [[ -z "${datastore_path}" ]]; then
        # No PGDATA configured; caller likely manages Postgres externally.
        return 0
    fi

    local datastore_clean_path
    datastore_clean_path="$(ini_get_value "${config_path}" "postgres" "datastore_clean")"
    if [[ -z "${datastore_clean_path}" ]]; then
        datastore_clean_path="${datastore_path}-clean"
    fi

    datastore_clean_path="$(resolve_datastore_path "${datastore_clean_path}")"

    local psql_config_path
    psql_config_path="$(ini_get_value "${config_path}" "postgres" "psql_config")"

    if [[ -z "${psql_config_path}" && -f "${BENCH_ROOT}/configs/postgresql.conf" ]]; then
        psql_config_path="configs/postgresql.conf"
    fi

    psql_config_path="$(resolve_path_under_root "${psql_config_path}")"

    local pg_log_path
    pg_log_path="$(ini_get_value "${config_path}" "postgres" "pg_log")"
    if [[ -z "${pg_log_path}" ]]; then
        pg_log_path="output/postgres.log"
    fi
    pg_log_path="$(resolve_path_under_root "${pg_log_path}")"

    local host
    host="$(ini_get_value "${config_path}" "postgres" "host")"
    local port
    port="$(ini_get_value "${config_path}" "postgres" "port")"
    local user
    user="$(ini_get_value "${config_path}" "postgres" "user")"

    # Ensure a clean snapshot exists (optionally create it).
    ensure_clean_snapshot \
        "${config_path}" \
        "${datastore_path}" \
        "${datastore_clean_path}" \
        "${psql_config_path}" \
        "${pg_log_path}" \
        "${host}" \
        "${port}" \
        "${user}" \
        "${aker_config_override}"

    log_info "Restoring clean snapshot for ${purpose} (no cache drop): ${datastore_clean_path} -> ${datastore_path}"

    pgctl_stop_force_all "${datastore_path}"

    rm -rf "${datastore_path}"
    cp -r "${datastore_clean_path}" "${datastore_path}"

    export_env_from_ini "${config_path}" "${aker_config_override}"

    pgctl_start "${datastore_path}" "${psql_config_path}" "${pg_log_path}"
    pg_wait_ready "${host}" "${port}" "${user}"
}

maybe_stop_postgres() {
    #
    # Stop PostgreSQL if we are managing PGDATA (postgres.datastore is configured).
    #
    # This helper is intentionally a no-op when postgres.datastore is empty, so that
    # callers can support externally managed PostgreSQL instances.
    #
    # Args:
    #   $1: config_path
    #
    local config_path="$1"

    if [[ -z "${config_path}" ]]; then
        log_warn "Config path is empty; skip maybe_stop_postgres"
        return 0
    fi

    local datastore_value
    datastore_value="$(ini_get_value "${config_path}" "postgres" "datastore")"
    datastore_value="$(resolve_datastore_path "${datastore_value}")"
    if [[ -z "${datastore_value}" ]]; then
        return 0
    fi

    pgctl_stop "${datastore_value}"
}


maybe_stop_postgres_graceful() {
    #
    # Graceful stop helper for success paths.
    # - Does NOT exit the script.
    # - Returns non-zero if pg_ctl stop fails for reasons other than "not running".
    #
    local config_path="$1"

    if [[ -z "${config_path}" ]]; then
        log_warn "Config path is empty; skip maybe_stop_postgres_graceful"
        return 0
    fi

    local datastore_value
    datastore_value="$(ini_get_value "${config_path}" "postgres" "datastore" || true)"
    datastore_value="$(resolve_datastore_path "${datastore_value}")"
    if [[ -z "${datastore_value}" ]]; then
        return 0
    fi

    pgctl_stop_graceful "${datastore_value}"
}

maybe_wait_for_aker_trace_export() {
    #
    # Aker trace export can be expensive. To avoid prematurely shutting down the server
    # (or container) before traces are written, optionally wait before stopping.
    #
    # Policy:
    # - If AK_BENCH_TRACE_EXPORT_WAIT_SEC is set, use it (0 disables).
    # - Otherwise, if AKER_CONFIG_PATH is set, default to 600 seconds.
    #
    local wait_sec="${AK_BENCH_TRACE_EXPORT_WAIT_SEC:-}"

    if [[ -z "${wait_sec}" ]]; then
        if [[ -n "${AKER_CONFIG_PATH:-}" ]]; then
            wait_sec="600"
        else
            wait_sec="0"
        fi
    fi

    if [[ "${wait_sec}" == "0" ]]; then
        return 0
    fi

    log_info "Waiting ${wait_sec} seconds for trace export before shutdown"
    sleep "${wait_sec}"
}

docker_cleanup_tmp_traces() {
    #
    # Remove trace artifacts under /tmp inside the container to keep it reusable.
    #
    if [[ -z "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
        return 0
    fi

    docker_exec_raw "\
        set -euo pipefail; \
        shopt -s nullglob; \
        rm -rf /tmp/aker_trace_* /tmp/topkache_trace_* /tmp/aker_*trace* /tmp/topkache_*trace* || true\
    "
}

maybe_shutdown_docker_container() {
    #
    # Stop the benchmark container if Docker mode is enabled.
    # By default we keep the container for reuse; removal is optional.
    #
    local config_path="$1"

    if [[ -z "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
        return 0
    fi

    require_cmd docker

    log_info "Stopping benchmark container (not removing image): ${AK_BENCH_DOCKER_CONTAINER_INTERNAL}"
    docker stop "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" >/dev/null 2>&1 || true

    if [[ "${AK_BENCH_DOCKER_REMOVE_CONTAINER_ON_EXIT_INTERNAL}" == "1" ]]; then
        log_warn "Removing container due to remove_container_on_exit=1: ${AK_BENCH_DOCKER_CONTAINER_INTERNAL}"
        docker rm "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" >/dev/null 2>&1 || true
    fi
}

capture_tmp_trace_list() {
    #
    # Capture current /tmp trace candidates into a sorted list file.
    #
    local out_path="$1"

    local tmp_file
    tmp_file="${out_path}.tmp"
    rm -f "${tmp_file}"

    if [[ -n "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
        docker_exec_raw "\
            set -euo pipefail; \
            shopt -s nullglob; \
            for p in /tmp/aker_trace_* /tmp/topkache_trace_* /tmp/aker_*trace* /tmp/topkache_*trace*; do \
                if [[ -e \"\${p}\" ]]; then printf '%s\\n' \"\${p}\"; fi; \
            done | sort -u\
        " > "${out_path}" || true

        if [[ ! -f "${out_path}" ]]; then
            : > "${out_path}"
        fi
        return 0
    fi

    # Host mode (legacy)
    shopt -s nullglob

    local patterns=(
        /tmp/aker_trace_*
        /tmp/topkache_trace_*
        /tmp/aker_*trace*
        /tmp/topkache_*trace*
    )

    local pattern
    for pattern in "${patterns[@]}"; do
        local p
        for p in ${pattern}; do
            if [[ -e "${p}" ]]; then
                printf "%s\n" "${p}" >> "${tmp_file}"
            fi
        done
    done

    shopt -u nullglob

    if [[ -f "${tmp_file}" ]]; then
        sort -u "${tmp_file}" > "${out_path}"
        rm -f "${tmp_file}"
    else
        : > "${out_path}"
    fi
}

collect_new_tmp_traces() {
    #
    # Compare /tmp trace candidates before/after and copy newly created ones to dest dirs.
    #
    # Args:
    #   $1: before_list_path
    #   $2: run_dest_dir (per-run)
    #   $3: merged_dest_dir (aggregated)
    #
    local before_list_path="$1"
    local run_dest_dir="$2"
    local merged_dest_dir="$3"

    mkdir -p "${run_dest_dir}"
    mkdir -p "${merged_dest_dir}"

    local after_list_path
    after_list_path="${before_list_path}.after"

    capture_tmp_trace_list "${after_list_path}"

    local new_list_path
    new_list_path="${before_list_path}.new"

    comm -13 "${before_list_path}" "${after_list_path}" > "${new_list_path}" || true

    while IFS= read -r path; do
        if [[ -z "${path}" ]]; then
            continue
        fi

        local base
        base="$(basename -- "${path}")"

        # Ensure unique names when copying.
        local run_target
        run_target="${run_dest_dir}/${base}"
        if [[ -e "${run_target}" ]]; then
            run_target="${run_dest_dir}/${base}_$(date +%s%N)"
        fi

        local merged_target
        merged_target="${merged_dest_dir}/${base}"
        if [[ -e "${merged_target}" ]]; then
            merged_target="${merged_dest_dir}/${base}_$(date +%s%N)"
        fi

        if [[ -n "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
            log_info "Collecting new container /tmp trace: ${path} -> ${run_target}"
            docker cp "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}:${path}" "${run_target}" >/dev/null 2>&1 || true
            docker cp "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}:${path}" "${merged_target}" >/dev/null 2>&1 || true
        else
            log_info "Collecting new /tmp trace: ${path} -> ${run_target}"
            cp -a "${path}" "${run_target}"
            cp -a "${path}" "${merged_target}"
        fi

    done < "${new_list_path}"

    rm -f "${after_list_path}" "${new_list_path}"
}

postgres_conf_get_value() {
    #
    # Read a simple "key = value" setting from postgresql.conf.
    # - Strips comments starting with '#'.
    # - Trims whitespace.
    # - Unquotes single/double quoted values.
    #
    # Args:
    #   $1: postgresql.conf path
    #   $2: key
    #
    local conf_path="$1"
    local key="$2"

    if [[ -z "${conf_path}" || ! -f "${conf_path}" || -z "${key}" ]]; then
        printf ""
        return 0
    fi

    awk -v key="${key}" '
        function ltrim(s) { sub(/^[ \t\r\n]+/, "", s); return s }
        function rtrim(s) { sub(/[ \t\r\n]+$/, "", s); return s }
        function trim(s)  { return rtrim(ltrim(s)) }

        {
            line = $0
            sub(/[ \t]*#.*/, "", line)
            line = trim(line)
            if (line == "") { next }

            pattern = "^" key "[ \t]*="
            if (line ~ pattern) {
                sub(pattern, "", line)
                line = trim(line)

                if (line ~ /^\x27.*\x27$/) {
                    sub(/^\x27/, "", line)
                    sub(/\x27$/, "", line)
                } else if (line ~ /^\x22.*\x22$/) {
                    sub(/^\x22/, "", line)
                    sub(/\x22$/, "", line)
                }

                print line
                exit
            }
        }
    ' "${conf_path}"
}

pgctl_stop_best_effort() {
    #
    # Best-effort pg_ctl stop used during cleanup.
    # This must NEVER exit the script.
    #
    local datastore_path="$1"

    if [[ -z "${datastore_path}" || ! -d "${datastore_path}" ]]; then
        log_warn "datastore path not found; skip best-effort pg_ctl stop: ${datastore_path}"
        return 0
    fi

    if [[ -n "${AK_BENCH_DOCKER_CONTAINER_INTERNAL}" ]]; then
        local stop_output
        stop_output="$(docker_exec_raw "pg_ctl -D '${datastore_path}' -m fast stop 2>&1" || true)"

        if [[ -z "${stop_output}" ]]; then
            return 0
        fi

        if [[ "${stop_output}" == *"no server running"* ]]; then
            log_warn "pg_ctl reports no server running (best-effort stop): ${datastore_path}"
            return 0
        fi

        # docker exec may fail if container is not running.
        if [[ "${stop_output}" == *"is not running"* || "${stop_output}" == *"No such container"* ]]; then
            log_warn "Docker container unavailable during best-effort pg_ctl stop: ${stop_output}"
            return 0
        fi

        # Keep going even if pg_ctl stop reports an error.
        log_warn "pg_ctl stop failed (best-effort, docker mode) for PGDATA=${datastore_path}: ${stop_output}"
        return 0
    fi

    if ! command -v pg_ctl >/dev/null 2>&1; then
        log_warn "pg_ctl not found; skip best-effort stop"
        return 0
    fi

    local stop_output
    stop_output="$(pg_ctl -D "${datastore_path}" -m fast stop 2>&1 || true)"

    if [[ -z "${stop_output}" ]]; then
        return 0
    fi

    if [[ "${stop_output}" == *"no server running"* ]]; then
        log_warn "pg_ctl reports no server running (best-effort stop): ${datastore_path}"
        return 0
    fi

    log_warn "pg_ctl stop failed (best-effort) for PGDATA=${datastore_path}: ${stop_output}"
    return 0
}

maybe_stop_postgres_best_effort() {
    #
    # Best-effort stop for cleanup paths.
    #
    # Args:
    #   $1: config_path
    #
    local config_path="$1"

    if [[ -z "${config_path}" ]]; then
        return 0
    fi

    local datastore_value
    datastore_value="$(ini_get_value "${config_path}" "postgres" "datastore" || true)"
    datastore_value="$(resolve_datastore_path "${datastore_value}")"
    if [[ -z "${datastore_value}" ]]; then
        return 0
    fi

    pgctl_stop_best_effort "${datastore_value}" || true
}

collect_postgres_logs() {
    #
    # Collect PostgreSQL logs for a benchmark run directory.
    # This copies:
    # - pg_ctl -l log (postgres.pg_log)
    # - logging_collector logs (log_directory from postgresql.conf, if under PGDATA)
    # - the postgresql.conf used
    # - postgresql.auto.conf (if present)
    #
    # Args:
    #   $1: config_path
    #   $2: run_dir
    #
    local config_path="$1"
    local run_dir="$2"

    if [[ -z "${config_path}" || -z "${run_dir}" || ! -d "${run_dir}" ]]; then
        return 0
    fi

    local datastore_value
    datastore_value="$(ini_get_value "${config_path}" "postgres" "datastore" || true)"
    local datastore_path
    datastore_path="$(resolve_datastore_path "${datastore_value}")"
    if [[ -z "${datastore_path}" ]]; then
        # Externally managed Postgres.
        return 0
    fi

    local dest_dir
    dest_dir="${run_dir}/postgres_logs"
    mkdir -p "${dest_dir}"

    # pg_ctl -l log file.
    local pg_log_value
    pg_log_value="$(ini_get_value "${config_path}" "postgres" "pg_log" || true)"
    if [[ -z "${pg_log_value}" ]]; then
        pg_log_value="output/postgres.log"
    fi
    local pg_log_path
    pg_log_path="$(resolve_path_under_root "${pg_log_value}")"

    if [[ -f "${pg_log_path}" ]]; then
        cp -a "${pg_log_path}" "${dest_dir}/pg_ctl.log" >/dev/null 2>&1 || true
    fi

    # postgresql.conf used.
    local psql_config_value
    psql_config_value="$(ini_get_value "${config_path}" "postgres" "psql_config" || true)"
    if [[ -z "${psql_config_value}" && -f "${BENCH_ROOT}/configs/postgresql.conf" ]]; then
        psql_config_value="configs/postgresql.conf"
    fi
    local psql_config_path
    psql_config_path="$(resolve_path_under_root "${psql_config_value}")"

    if [[ -f "${psql_config_path}" ]]; then
        cp -a "${psql_config_path}" "${dest_dir}/postgresql.conf" >/dev/null 2>&1 || true
    fi

    # postgresql.auto.conf is generated by ALTER SYSTEM.
    if [[ -f "${datastore_path}/postgresql.auto.conf" ]]; then
        cp -a "${datastore_path}/postgresql.auto.conf" "${dest_dir}/postgresql.auto.conf" >/dev/null 2>&1 || true
    fi

    # logging_collector logs.
    local log_dir_value
    log_dir_value="$(postgres_conf_get_value "${psql_config_path}" "log_directory" || true)"

    local collector_dir=""
    if [[ -n "${log_dir_value}" ]]; then
        if [[ "${log_dir_value}" == /* ]]; then
            collector_dir="${log_dir_value}"
        else
            collector_dir="${datastore_path}/${log_dir_value}"
        fi
    else
        # Heuristic fallback for common defaults.
        if [[ -d "${datastore_path}/log" ]]; then
            collector_dir="${datastore_path}/log"
        elif [[ -d "${datastore_path}/logs" ]]; then
            collector_dir="${datastore_path}/logs"
        fi
    fi

    if [[ -n "${collector_dir}" && -d "${collector_dir}" ]]; then
        rm -rf "${dest_dir}/collector" >/dev/null 2>&1 || true
        cp -a "${collector_dir}" "${dest_dir}/collector" >/dev/null 2>&1 || true
    fi
}
