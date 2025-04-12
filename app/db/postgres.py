from app.config import DB_CONNECT_TIMEOUT, DB_HOST, DB_PASSWORD, DB_PORT, DB_USER
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# -- Bronze DB --
BRONZE_DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/bronze"
bronze_engine = create_engine(BRONZE_DB_URL, connect_args={"connect_timeout": DB_CONNECT_TIMEOUT})
# autocommit and authoflush set to false to ensure atomicity
BronzeSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=bronze_engine)

# -- Silver DB --
SILVER_DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/silver"
silver_engine = create_engine(SILVER_DB_URL, connect_args={"connect_timeout": DB_CONNECT_TIMEOUT})
# autocommit and authoflush set to false to ensure atomicity
SilverSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=silver_engine)

# -- Gold DB --
GOLD_DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/gold"
gold_engine = create_engine(GOLD_DB_URL, connect_args={"connect_timeout": DB_CONNECT_TIMEOUT})
# autocommit and authoflush set to false to ensure atomicity
GoldSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=gold_engine)


## -- Create all tables if models are defined --
def init_all_databases():
    from app.schemas import bronze, silver, gold

    bronze.Base.metadata.create_all(bind=bronze_engine)
    silver.Base.metadata.create_all(bind=silver_engine)
    gold.Base.metadata.create_all(bind=gold_engine)