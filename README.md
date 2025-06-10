# DEPLOY

## Local testing
Localy run.
Remove references to `pysqlite3-binary` from the code if needed.

```
streamlit run main.py
```

## Streamlit deploy

First, check if `pysql ite3-binary` is present in requirements.txt. It is needed for the remote deploy. 
You should also see the following code on top of main.py:

```
 __import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

1. Push changes to branck `iaris`
2. Login in Streamlit with Github and refresh deploy.

