from pydantic import BaseModel
from typing import Optional


class RadialSymmetryFunctionConfig(BaseModel):
    bond: Optional[str]=None
    eta: float
    rs: float
    rcut: float


class AngularSymmetryFunctionConfig(BaseModel):
    bond: Optional[str]=None
    lambdas: float
    zeta: float
    eta: float
    rcut: float
    