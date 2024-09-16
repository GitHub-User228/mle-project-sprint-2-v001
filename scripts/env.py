from pathlib import Path, PosixPath

from pydantic import BaseSettings, validator


class EnviromentVariables(BaseSettings):
    """
    Defines an `EnviromentVariables` class that loads and validates
    environment variables that are used in the MlFlow project.
    """

    db_destination_host: str
    db_destination_port: int
    db_destination_name: str
    db_destination_user: str
    db_destination_password: str

    s3_bucket_name: str
    aws_access_key_id: str
    aws_secret_access_key: str

    mlflow_s3_endpoint_url: str
    tracking_server_host: str
    tracking_server_port: int

    log_dir: PosixPath = Path("../logs")
    config_dir: PosixPath = Path("../config")

    class Config:
        env_file = ".env"

    @validator("*")
    def check_variable(cls, v: str | PosixPath, field: str) -> str | PosixPath:
        """
        Validates and checks the environment variable specified by the
        `field` parameter. Returns the validated variable.

        Args:
            v (str | PosixPath):
                The environment variable value to be validated.
            field (str):
                The name of the environment variable.

        Raises:
            ValueError:
                If the environment variable is not set.

        Returns:
            str | PosixPath:
                The validated environment variable value.
        """
        if v is None:
            raise ValueError(
                f"Environment variable '{field.name}' is not set."
            )
        if isinstance(v, PosixPath):
            if not v.exists():
                v.mkdir(parents=True, exist_ok=True)
        return v

    @classmethod
    def from_env(cls) -> "EnviromentVariables":
        """
        Create a new instance of EnviromentVariables by loading
        environment variables.

        Returns:
            EnviromentVariables:
                A new instance of EnviromentVariables with loaded and
                validated environment variables.

        Raises:
            ValueError:
                If any required environment variable is not set.
        """
        return cls()


env_vars = EnviromentVariables.from_env()
