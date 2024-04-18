import json
import subprocess

MOSAIC_API_KEY = "gLg6bt.L-cyANTkD/o0lWDCIHM9N8ZbdCp3LygiX7U0T7cR"
ACCESS_TOKEN_FILENAME = "access_token.json"
SERVICE_ACCOUNT = "mosaic-api@etsy-mlinfra-dev.iam.gserviceaccount.com"

def get_and_save_access_token(service_account, filename=ACCESS_TOKEN_FILENAME):
    """Get the access token for the given service account using gcloud."""
    command = [
        "gcloud",
        "auth",
        "print-access-token",
        "--impersonate-service-account",
        service_account,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    token = result.stdout.strip()
    # Save the access token to a JSON file.
    with open(filename, "w") as file:
        json.dump({"access_token": token}, file)


def setup_mcli():
    """Setup the Mosaic CLI with mcli set api-key <your api-key>"""
    command = ["mcli", "set", "api-key", MOSAIC_API_KEY]
    subprocess.run(command, check=True)
    # run mcli config to get response and print
    command = ["mcli", "config"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    print(f"Mosaic CLI setup complete. mcli config: \n{result.stdout}")


def create_secret_docker(filename=ACCESS_TOKEN_FILENAME):
    """Run the CLI command to create a docker secret."""
    with open(filename) as file:
        password = json.load(file)["access_token"]
    command = [
        "mcli",
        "create",
        "secret",
        "docker",
        "--name",
        "etsy-docker",
        "--username",
        "oauth2accesstoken",
        "--password",
        password,
        "--server",
        "us-central1-docker.pkg.dev",
    ]
    subprocess.run(command, check=True)


def create_secret_env(filename=ACCESS_TOKEN_FILENAME):
    """Run the CLI command to create an environment secret with the access token."""
    with open(filename) as file:
        token = json.load(file)["access_token"]
        token_arg = f"GC_ACCESS_TOKEN={token}"

    command = ["mcli", "create", "secret", "env", token_arg]
    subprocess.run(command, check=True)

def run_mcli():
    """Run the CLI command to create a run with mcli run -f job.yaml --follow"""
    command = ["mcli", "run", "-f", "nir-run.yaml", "--follow"]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    setup_mcli()
    # Obtain the access token and save
    token = get_and_save_access_token(SERVICE_ACCOUNT)
    # Execute CLI commands
    create_secret_docker()
    create_secret_env()
    run_mcli()