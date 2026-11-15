"""equation counter: the highest equation number given in a conversation

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Existing conversations start at 0: their messages are numbered when they are next read or continued.
    op.add_column(
        "conversations",
        sa.Column("equation_counter", sa.Integer(), server_default="0", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("conversations", "equation_counter")
