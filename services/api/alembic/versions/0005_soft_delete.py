"""soft delete for calculations and derivations (undo)

Revision ID: 0005
Revises: 0004
"""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("calculations") as batch:
        batch.add_column(sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
        batch.create_index("ix_calculations_deleted_at", ["deleted_at"])
    with op.batch_alter_table("derivations") as batch:
        batch.add_column(sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("derivations") as batch:
        batch.drop_column("deleted_at")
    with op.batch_alter_table("calculations") as batch:
        batch.drop_index("ix_calculations_deleted_at")
        batch.drop_column("deleted_at")
