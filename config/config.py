# -*- coding:utf-8 -*-
# author: Xinge

from pathlib import Path

from strictyaml import Bool, Float, Int, Map, Seq, Str, as_document, load

model_params = Map(
    {
        "model_architecture": Str(),
        "output_shape": Seq(Int()),
        "fea_dim": Int(),
        "out_fea_dim": Int(),
        "num_class": Int(),
        "num_input_features": Int(),
        "use_norm": Bool(),
        "init_size": Int(),
    }
)

dataset_params = Map(
    {
        "dataset_type": Str(),
        "pc_dataset_type": Str(),
        "ignore_label": Int(),
        "return_test": Bool(),
        "fixed_volume_space": Bool(),
        "label_mapping": Str(),
        "max_volume_space": Seq(Float()),
        "min_volume_space": Seq(Float()),
    }
)


train_data_loader = Map(
    {
        "data_path": Str(),
        "imageset": Str(),
        "return_ref": Bool(),
        "batch_size": Int(),
        "shuffle": Bool(),
        "num_workers": Int(),
    }
)

val_data_loader = Map(
    {
        "data_path": Str(),
        "imageset": Str(),
        "return_ref": Bool(),
        "batch_size": Int(),
        "shuffle": Bool(),
        "num_workers": Int(),
    }
)

train_params = Map(
    {
        "model_load_path": Str(),
        "model_save_path": Str(),
        "checkpoint_every_n_steps": Int(),
        "max_num_epochs": Int(),
        "eval_every_n_steps": Int(),
        "learning_rate": Float()
     }
)

schema_v4 = Map(
    {
        "format_version": Int(),
        "model_params": model_params,
        "dataset_params": dataset_params,
        "train_data_loader": train_data_loader,
        "val_data_loader": val_data_loader,
        "train_params": train_params,
    }
)


SCHEMA_FORMAT_VERSION_TO_SCHEMA = {4: schema_v4}


def load_config_data(path: str) -> dict:
    yaml_string = Path(path).read_text()
    cfg_without_schema = load(yaml_string, schema=None)
    schema_version = int(cfg_without_schema["format_version"])
    if schema_version not in SCHEMA_FORMAT_VERSION_TO_SCHEMA:
        raise Exception(f"Unsupported schema format version: {schema_version}.")

    strict_cfg = load(yaml_string, schema=SCHEMA_FORMAT_VERSION_TO_SCHEMA[schema_version])
    cfg: dict = strict_cfg.data
    return cfg


def config_data_to_config(data):  # type: ignore
    return as_document(data, schema_v4)


def save_config_data(data: dict, path: str) -> None:
    cfg_document = config_data_to_config(data)
    with open(Path(path), "w") as f:
        f.write(cfg_document.as_yaml())
