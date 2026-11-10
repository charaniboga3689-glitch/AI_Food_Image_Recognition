import sys
import os
from streamlit.web import cli as stcli

if __name__ == "__main__":
    streamlit_app_path = os.path.join(os.path.dirname(__file__), "temp_repo", "app.py")
    sys.argv = ["streamlit", "run", streamlit_app_path, "--global.developmentMode=false"]
    sys.exit(stcli.main())
