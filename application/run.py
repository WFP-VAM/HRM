import os
from app import application

print("* Flask starting server...")

# Bind to PORT if defined, otherwise default to 5000.
port = int(os.environ.get('PORT', 5001))
application.run(host='0.0.0.0', port=port)
