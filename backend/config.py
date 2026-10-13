import os
import secrets


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))


def load_optional_dotenv():
	"""Load KEY=VALUE pairs from a local .env without overriding real environment variables."""
	path = os.path.join(ROOT_DIR, ".env")
	if not os.path.isfile(path):
		return
	with open(path, encoding="utf-8") as handle:
		for raw_line in handle:
			line = raw_line.strip()
			if not line or line.startswith("#") or "=" not in line:
				continue
			key, value = line.split("=", 1)
			key = key.strip()
			value = value.strip().strip("'").strip('"')
			if key and key not in os.environ:
				os.environ[key] = value


load_optional_dotenv()


def _signing_key_from_env(name):
	value = os.environ.get(name)
	if value:
		return value
	if os.environ.get("DEALORA_ENV", "development").lower() == "production":
		raise RuntimeError("{} must be set in production".format(name))
	return secrets.token_urlsafe(32)


class Config:
	SECRET_KEY = _signing_key_from_env("DEALORA_SECRET_KEY")
	JWT_SECRET_KEY = _signing_key_from_env("DEALORA_JWT_SECRET_KEY")
	SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(BASE_DIR, "dealora.db")
	SQLALCHEMY_TRACK_MODIFICATIONS = False
	JWT_ACCESS_TOKEN_EXPIRES = 60 * 60 * 12
	CORS_ORIGINS = os.environ.get("DEALORA_CORS_ORIGINS", "*")
