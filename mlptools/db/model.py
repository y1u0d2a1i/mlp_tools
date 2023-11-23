from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base


Base = declarative_base()

class Structure(Base):
    __tablename__ = 'structure'
    id = Column(String, primary_key=True)
    original_path = Column(String, unique=True)
    structure_id = Column(Integer)
    structure_name = Column(String)
    calculation_type = Column(String)