"""Flask extension instances for deferred initialization.

Extensions are created here and configured later via init_app() in the
application factory.
"""

from flask_caching import Cache

# Object created but not yet configured (init_app happens in create_app)
cache = Cache()
