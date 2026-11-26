"""Initial database schema for SIGNBRIDGE AI

Revision ID: 0001_initial_schema
Revises: 
Create Date: 2026-11-26 12:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '0001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='active'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)

    # 2. Create translation_history table
    op.create_table(
        'translation_history',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('user_id', sa.String(length=36), nullable=False),
        sa.Column('source_language', sa.String(length=20), nullable=False, server_default='ISL'),
        sa.Column('translation_direction', sa.String(length=30), nullable=False, server_default='sign_to_text'),
        sa.Column('input_text', sa.Text(), nullable=True),
        sa.Column('recognized_output', sa.Text(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('recognition_path', sa.String(length=30), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_translation_history_id'), 'translation_history', ['id'], unique=False)
    op.create_index(op.f('ix_translation_history_user_id'), 'translation_history', ['user_id'], unique=False)
    op.create_index(op.f('ix_translation_history_direction'), 'translation_history', ['translation_direction'], unique=False)
    op.create_index(op.f('ix_translation_history_created_at'), 'translation_history', ['created_at'], unique=False)

    # 3. Create learning_lessons table
    op.create_table(
        'learning_lessons',
        sa.Column('id', sa.String(length=50), nullable=False),
        sa.Column('title', sa.String(length=150), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('difficulty', sa.String(length=20), nullable=False, server_default='beginner'),
        sa.Column('category', sa.String(length=50), nullable=False, server_default='General'),
        sa.Column('content_metadata', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_learning_lessons_id'), 'learning_lessons', ['id'], unique=False)
    op.create_index(op.f('ix_learning_lessons_difficulty'), 'learning_lessons', ['difficulty'], unique=False)
    op.create_index(op.f('ix_learning_lessons_category'), 'learning_lessons', ['category'], unique=False)

    # 4. Create learning_progress table
    op.create_table(
        'learning_progress',
        sa.Column('id', sa.String(length=36), nullable=False),
        sa.Column('user_id', sa.String(length=36), nullable=False),
        sa.Column('lesson_id', sa.String(length=50), nullable=False),
        sa.Column('completion_status', sa.String(length=20), nullable=False, server_default='not_started'),
        sa.Column('score', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('last_activity', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['lesson_id'], ['learning_lessons.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'lesson_id', name='uq_user_lesson_progress')
    )
    op.create_index(op.f('ix_learning_progress_id'), 'learning_progress', ['id'], unique=False)
    op.create_index(op.f('ix_learning_progress_user_id'), 'learning_progress', ['user_id'], unique=False)
    op.create_index(op.f('ix_learning_progress_lesson_id'), 'learning_progress', ['lesson_id'], unique=False)
    op.create_index(op.f('ix_learning_progress_status'), 'learning_progress', ['completion_status'], unique=False)


def downgrade() -> None:
    op.drop_table('learning_progress')
    op.drop_table('learning_lessons')
    op.drop_table('translation_history')
    op.drop_table('users')
