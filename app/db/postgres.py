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

        self._engine = None
        self._sessionmaker = None

    def build_url(self, async_mode=True):
        """Constructs the db URL using either asyncpg driver (for async code like FastAPI)
        or psycopg2 derive (for synchronous tools like Alembic)."""
        driver = "postgresql+asyncpg" if async_mode else "postgresql+psycopg2"
        return f"{driver}://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}"

    def create_engine(self):
        """
        Creates the SQLAlchemy async engine for the database if the engine hasn't been created yet.

        Returns:
            AsyncEngine: A SQLAlchemy async engine connected to the configured database.

        Notes:
            - Uses the `asyncpg` driver for async support.
            - The engine is created only once and reused for subsequent calls.
            - Connection timeout is configured via `DB_CONNECT_TIMEOUT`.
            - The engine is created with `echo=False` to disable SQL statement logging.

        """
        if self._engine is None:
            self._engine = create_async_engine(
                self.build_url(async_mode=True),
                echo=False,
                connect_args={"timeout": DB_CONNECT_TIMEOUT},
            )
        return self._engine

    def create_session(self):
        """
        Creates the SQLAlchemy async sessionmaker if it hasn't been created yet.

        Returns:
            sessionmaker: A factory for creating AsyncSession instances.

        Notes:
            - Behaves like a singleton: only one sessionmaker is created per instance.
            - The sessionmaker is bound to the async engine created by `create_engine()`.
            - `expire_on_commit=False` keeps ORM objects usable after commit.
            - Usage: `async with db.create_session() as session:`
        """
        if self._sessionmaker is None:
            self._sessionmaker = sessionmaker(
                bind=self.create_engine(),
                class_=AsyncSession,
                expire_on_commit=False,
            )
        return self._sessionmaker
