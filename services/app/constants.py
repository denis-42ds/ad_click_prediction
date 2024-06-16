from pydantic import BaseModel

REQUIRED_MODEL_PARAMS=['region_id',
    'city_id',
    'tags_cont',
    'tags_bhv',
    'rubrica',
    'rate',
    'ctr_sort',
    'rv_perc',
    'slider']

MODEL_PATH='model.pkl'

class ModelParams(BaseModel):
    region_id: float = 38.0
    city_id: float = 590.0
    tags_cont: float = 0.0
    tags_bhv: float = 0.5
    rubrica: float = 1.812
    rate: float = 1.7
    ctr_sort: float = 1.3072
    rv_perc: float = 66.41
    slider: float = 1.0
    mean_rate_by_city: float = 3.615865
    mean_rate_by_region: float = 3.558722
    mean_rv_perc_by_city: float = 36.435853
    mean_rv_perc_by_region: float = 36.044791
    mean_ctr_sort_by_city: float = 0.986916
    mean_ctr_sort_by_region: float = 0.995134


