from ml_engine.train_model import add

def test_addition():
  result=add(2,3)
  assert result==5
