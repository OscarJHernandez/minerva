"""Workbench derivations: ordered, linked calculations.

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-12
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "derivations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "user_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_derivations_user_updated", "derivations", ["user_id", "updated_at"])
    with op.batch_alter_table("calculations") as batch:
        batch.add_column(sa.Column("derivation_id", sa.Uuid(), nullable=True))
        batch.add_column(sa.Column("parent_id", sa.Uuid(), nullable=True))
        batch.create_foreign_key(
            "fk_calculations_derivation_id",
            "derivations",
            ["derivation_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch.create_foreign_key(
            "fk_calculations_parent_id",
            "calculations",
            ["parent_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch.create_index("ix_calculations_derivation_id", ["derivation_id"])


def downgrade() -> None:
    with op.batch_alter_table("calculations") as batch:
        batch.drop_index("ix_calculations_derivation_id")
        batch.drop_constraint("fk_calculations_parent_id", type_="foreignkey")
        batch.drop_constraint("fk_calculations_derivation_id", type_="foreignkey")
        batch.drop_column("parent_id")
        batch.drop_column("derivation_id")
    op.drop_table("derivations")
