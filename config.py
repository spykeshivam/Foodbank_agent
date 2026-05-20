import base64
import os
import tempfile

# Credentials — on Render set GOOGLE_CREDENTIALS_B64 to the base64-encoded
# contents of credentials.json. Locally the file is read directly.
_creds_b64 = os.getenv("GOOGLE_CREDENTIALS_B64")
if _creds_b64:
    # Add padding in case it was stripped during copy-paste
    _creds_b64 += "=" * (-len(_creds_b64) % 4)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="wb") as _tmp:
        _tmp.write(base64.b64decode(_creds_b64))
    CREDENTIALS_FILE = _tmp.name
else:
    CREDENTIALS_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials.json")

# Scopes
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

# Cache Settings
CACHE_TYPE = "SimpleCache"
CACHE_DEFAULT_TIMEOUT = 300
