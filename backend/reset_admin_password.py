"""Set a new password for an existing Dealora user (default: admin@dealora.com).

Usage (PowerShell, from the project folder, with the virtual environment active):
    $env:DEALORA_NEW_PASSWORD = "choose-a-password-12-chars-or-more"
    python backend\\reset_admin_password.py

Optional: $env:DEALORA_RESET_EMAIL = "another@email.com"
The password is read from the environment so it is never stored in the code.
"""
import os

from app import create_app
from models import User, db

email = os.environ.get("DEALORA_RESET_EMAIL", "admin@dealora.com")
new_password = os.environ.get("DEALORA_NEW_PASSWORD", "")

if len(new_password) < 12:
	raise SystemExit("Set DEALORA_NEW_PASSWORD to a password of at least 12 characters.")

app = create_app()
with app.app_context():
	user = User.query.filter_by(email=email).first()
	if user is None:
		raise SystemExit(
			"No user with email {}. Run backend\\seed.py first to create the demo users.".format(email)
		)
	user.set_password(new_password)
	db.session.commit()
	print("Password updated for {} (role: {}).".format(email, user.role))