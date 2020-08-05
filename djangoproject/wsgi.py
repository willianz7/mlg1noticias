"""
WSGI config for djangoproject project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/dev/howto/deployment/wsgi/
"""

import os, sys

sys.path.append('/home/willian/django-dev-ml/djangoproject')

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'djangoproject.settings')

application = get_wsgi_application()

