import os
import base64
import tempfile

from dotenv import load_dotenv
load_dotenv()  # loads .env if present (local dev); no-op in production

# Credentials — on Render set GOOGLE_CREDENTIALS_B64 to the base64-encoded
# contents of credentials.json. Locally the file is read directly.
_creds_b64 = os.getenv("GOOGLE_CREDENTIALS_B64")
if _creds_b64:
    # Add padding in case it was stripped during copy-paste
    _creds_b64 += "=" * (-len(_creds_b64) % 4)
    _tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="wb")
    _tmp.write(base64.b64decode(_creds_b64))
    _tmp.close()
    CREDENTIALS_FILE = _tmp.name
else:
    CREDENTIALS_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")

# Scopes
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# Cache Settings
CACHE_TYPE = "SimpleCache"
CACHE_DEFAULT_TIMEOUT = 300
