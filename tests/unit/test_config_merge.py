"""Config Deep Merge 테스트"""

from pathlib import Path

from weighted_mtp.cli.train import deep_merge, load_config


class TestDeepMerge:
    """deep_merge() 함수 테스트"""

    def test_simple_override(self):
        """단순 값 override"""
        base = {"a": 1, "b": 2}
        override = {"a": 10}
        result = deep_merge(base, override)

        assert result == {"a": 10, "b": 2}

    def test_nested_merge(self):
        """중첩 dict 병합"""
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 10}, "e": 4}
        result = deep_merge(base, override)

        assert result == {"a": {"b": 10, "c": 2}, "d": 3, "e": 4}

    def test_deep_nested_merge(self):
        """깊은 중첩 dict 병합"""
        base = {
            "training": {
                "stage1": {"n_epochs": 0.5, "learning_rate": 1e-4},
                "stage2": {"n_epochs": 2.5, "beta": 0.9, "value_coef": 0.5},
            }
        }
        override = {"training": {"stage2": {"beta": 1.2}}}
        result = deep_merge(base, override)

        # stage2.beta만 override, 나머지는 유지
        assert result["training"]["stage2"]["beta"] == 1.2
        assert result["training"]["stage2"]["n_epochs"] == 2.5
        assert result["training"]["stage2"]["value_coef"] == 0.5
        assert result["training"]["stage1"]["n_epochs"] == 0.5

    def test_new_key_addition(self):
        """새로운 키 추가"""
        base = {"a": 1}
        override = {"b": 2}
        result = deep_merge(base, override)

        assert result == {"a": 1, "b": 2}

    def test_replace_non_dict_value(self):
        """dict가 아닌 값 교체"""
        base = {"a": {"b": 1}}
        override = {"a": [1, 2, 3]}  # dict → list 교체
        result = deep_merge(base, override)

        assert result == {"a": [1, 2, 3]}

    def test_empty_override(self):
        """빈 override"""
        base = {"a": 1, "b": 2}
        override = {}
        result = deep_merge(base, override)

        assert result == {"a": 1, "b": 2}

    def test_empty_base(self):
        """빈 base"""
        base = {}
        override = {"a": 1, "b": 2}
        result = deep_merge(base, override)

        assert result == {"a": 1, "b": 2}

    def test_original_not_modified(self):
        """원본 dict가 수정되지 않는지 확인"""
        base = {"a": {"b": 1}}
        override = {"a": {"b": 10}}
        result = deep_merge(base, override)

        # 원본 유지
        assert base == {"a": {"b": 1}}
        assert result == {"a": {"b": 10}}


class TestLoadConfigWithMerge:
    """load_config() Deep merge 통합 테스트"""

    def test_defaults_only(self, project_root: Path):
        """defaults.yaml만 로딩"""
        config_path = project_root / "configs" / "defaults.yaml"
        config = load_config(config_path)

        assert "project" in config
        assert "training" in config
        assert config["training"]["stage2"]["beta"] == 0.9

    def test_recipe_override_partial(self, project_root: Path):
        """Recipe가 일부만 override"""
        config_path = project_root / "configs" / "defaults.yaml"
        recipe_path = project_root / "configs" / "recipe.verifiable.yaml"
        config = load_config(config_path, recipe_path)

        # Recipe의 override 반영
        assert "experiment" in config
        assert config["experiment"]["name"] == "verifiable-critic-wmtp"

        # Recipe에서 정의한 training.stage2.beta (있으면 override, 없으면 defaults 유지)
        if "training" in config and "stage2" in config["training"]:
            assert "beta" in config["training"]["stage2"]

        # Recipe에 없는 project는 defaults 유지
        assert "project" in config
        assert config["project"]["name"] == "weighted-mtp"

    def test_realistic_scenario(self, project_root: Path):
        """실제 사용 시나리오: defaults + recipe 병합"""
        import yaml

        # Defaults 확인
        with open(project_root / "configs" / "defaults.yaml") as f:
            defaults = yaml.safe_load(f)

        # Recipe 확인
        with open(project_root / "configs" / "recipe.verifiable.yaml") as f:
            recipe = yaml.safe_load(f)

        # Load config (deep merge)
        config = load_config(
            project_root / "configs" / "defaults.yaml",
            project_root / "configs" / "recipe.verifiable.yaml",
        )

        # Defaults의 project는 유지
        assert config["project"] == defaults["project"]

        # Recipe의 experiment는 추가
        assert config["experiment"] == recipe["experiment"]

        # Training은 병합 (defaults + recipe)
        if "training" in defaults and "training" in recipe:
            # Recipe에서 override한 값
            for key in recipe["training"]:
                if isinstance(recipe["training"][key], dict):
                    # 중첩 dict는 deep merge
                    for subkey in recipe["training"][key]:
                        assert (
                            config["training"][key][subkey]
                            == recipe["training"][key][subkey]
                        )


class TestConfigMergeEdgeCases:
    """Edge case 테스트"""

    def test_list_override(self):
        """리스트는 병합이 아니라 교체"""
        base = {"a": [1, 2, 3]}
        override = {"a": [4, 5]}
        result = deep_merge(base, override)

        assert result == {"a": [4, 5]}  # 교체됨

    def test_null_value_override(self):
        """None 값 override"""
        base = {"a": 1}
        override = {"a": None}
        result = deep_merge(base, override)

        assert result == {"a": None}

    def test_deep_three_levels(self):
        """3단계 중첩"""
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"c": 10}}}
        result = deep_merge(base, override)

        assert result == {"a": {"b": {"c": 10, "d": 2}}}
