from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from mlptools.db.model import Base


db_location_dict = {
    "Si": "/Users/y1u0d2/desktop/Lab/data/qe_data/Si",
    "SiO2": "/Users/y1u0d2/desktop/Lab/data/qe_data/SiO2",
}

def get_engine(path2db):
    engine = create_engine(f'sqlite:///{path2db}/structure.db')
    Base.metadata.create_all(engine)
    return engine


def get_session(structure_symbol:str):
    if db_location_dict.get(structure_symbol) is None:
        raise ValueError(f"structure_symbol {structure_symbol} not in db_location_dict")
    
    engine = get_engine(db_location_dict[structure_symbol])
    Session = sessionmaker(bind=engine)
    session = Session()
    return session