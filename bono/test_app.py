import pytest
from bono.app import app as flask_app
import pandas as pd

@pytest.fixture()
def client():
    flask_app.config.update({"TESTING": True})
    with flask_app.test_client() as client:
        yield client

def test_hello(client):

   # df = pd.read_csv("inferenza.csv")
    input_data = {
    "fixed_acidity":7.6,
    "residual_sugar":14.3,
    "alcohol":10.1,
    "density":0.9924,
    "quality_label":"high"
    }

    response = client.post("/infer",json = {"data": input_data})
    
    assert response.status_code == 200
    data = response.get_json()
    assert data["predictions"] == "high"