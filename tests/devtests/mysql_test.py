import optuna


storage = "mysql://hpvsim_user@localhost/hpvsim_db"
name = "distributed-example"

def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2

if __name__ == "__main__":
    optuna.create_study(storage=storage, study_name=name)
    study = optuna.load_study(study_name=name, storage=storage)
    study.optimize(objective, n_trials=100)