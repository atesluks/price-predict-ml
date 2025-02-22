"""Init migration

Revision ID: 488335177a33
Revises: 
Create Date: 2025-01-22 18:41:38.792192

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '488335177a33'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('cache_price',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('model_name', sa.String(), nullable=False),
    sa.Column('price', sa.Float(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_cache_price_id'), 'cache_price', ['id'], unique=False)
    op.create_table('data_cg_coins_market_chart_1d',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('pair', sa.String(), nullable=False),
    sa.Column('time', sa.DateTime(), nullable=False),
    sa.Column('price', sa.Float(), nullable=False),
    sa.Column('mcap', sa.Float(), nullable=False),
    sa.Column('volume', sa.Float(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_data_cg_coins_market_chart_1d_id'), 'data_cg_coins_market_chart_1d', ['id'], unique=False)
    op.create_table('data_cg_coins_market_chart_1h',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('pair', sa.String(), nullable=False),
    sa.Column('time', sa.DateTime(), nullable=False),
    sa.Column('price', sa.Float(), nullable=False),
    sa.Column('mcap', sa.Float(), nullable=False),
    sa.Column('volume', sa.Float(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_data_cg_coins_market_chart_1h_id'), 'data_cg_coins_market_chart_1h', ['id'], unique=False)
    op.create_table('logs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('date', sa.DateTime(), nullable=False),
    sa.Column('type', sa.String(), nullable=False),
    sa.Column('emitter', sa.String(), nullable=False),
    sa.Column('message', sa.String(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_logs_id'), 'logs', ['id'], unique=False)
    op.create_table('model_training_queue',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('model_name', sa.String(), nullable=False),
    sa.Column('added_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.Column('data_worker', sa.String(), nullable=False),
    sa.Column('status', sa.Enum('PENDING', 'TRAINING', 'DONE', name='model_training_status'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('model_training_queue')
    op.drop_index(op.f('ix_logs_id'), table_name='logs')
    op.drop_table('logs')
    op.drop_index(op.f('ix_data_cg_coins_market_chart_1h_id'), table_name='data_cg_coins_market_chart_1h')
    op.drop_table('data_cg_coins_market_chart_1h')
    op.drop_index(op.f('ix_data_cg_coins_market_chart_1d_id'), table_name='data_cg_coins_market_chart_1d')
    op.drop_table('data_cg_coins_market_chart_1d')
    op.drop_index(op.f('ix_cache_price_id'), table_name='cache_price')
    op.drop_table('cache_price')
    # ### end Alembic commands ###
