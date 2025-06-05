from config import DB_CONNECT_TIMEOUT
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker


class AsyncPostgresDB:
    def __init__(self, user, password, host, port, db_name):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.db_name = db_name
        self.port = int(port)

    def build_url(self, async_mode=True):
        """Constructs the db URL using either asyncpg driver (for async code like FastAPI)
        or psycopg2 derive (for synchronous tools like Alembic)."""
        driver = "postgresql+asyncpg" if async_mode else "postgresql+psycopg2"
        return f"{driver}://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}"

    def create_engine(self):
        """Creates and returns an SQLAlchemy async engine for the database."""
        url = self.build_url(async_mode=True)
        return create_async_engine(
            url, echo=False, connect_args={"timeout": DB_CONNECT_TIMEOUT}
        )

    def create_session(self):
        """Initializes and returns an async sessionmaker for the database engine."""
        engine = self.create_engine()
        return sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
