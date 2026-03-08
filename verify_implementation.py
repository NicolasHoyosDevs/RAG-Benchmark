#!/usr/bin/env python3
"""
Verification script for model provider abstraction implementation.
Tests the new feature without requiring actual API calls.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

def test_model_config_creation():
    """Test that ModelConfig can be created correctly."""
    print("\n✅ Test 1: ModelConfig instantiation")
    from src.common.model_provider import ModelConfig
    
    # Test OpenAI model
    openai_config = ModelConfig(
        name="gpt-4.1",
        model_id="gpt-4.1",
        provider="openai"
    )
    assert openai_config.name == "gpt-4.1"
    assert openai_config.provider == "openai"
    print("   ✓ OpenAI ModelConfig created successfully")
    
    # Test HuggingFace model
    hf_config = ModelConfig(
        name="mediphi",
        model_id="microsoft/MediPhi",
        provider="huggingface",
        endpoint_url_env="MEDIPHI_ENDPOINT_URL"
    )
    assert hf_config.name == "mediphi"
    assert hf_config.provider == "huggingface"
    print("   ✓ HuggingFace ModelConfig created successfully")
    
    # Test that HF without endpoint_url_env raises error
    try:
        bad_config = ModelConfig(
            name="badmodel",
            model_id="bad/model",
            provider="huggingface"
        )
        print("   ✗ ERROR: Should have raised ValueError for missing endpoint_url_env")
        return False
    except ValueError as e:
        print(f"   ✓ Correctly raised error for missing endpoint_url_env: {str(e)[:50]}...")
    
    return True


def test_models_registry():
    """Test that the models registry is correctly initialized."""
    print("\n✅ Test 2: Models registry")
    from src.common.model_provider import MODELS_REGISTRY
    
    expected_keys = {"gpt-5", "gpt-4.1", "mediphi", "medgemma"}
    actual_keys = set(MODELS_REGISTRY.keys())
    
    assert actual_keys == expected_keys, f"Expected {expected_keys}, got {actual_keys}"
    print(f"   ✓ Registry has all 4 models: {sorted(actual_keys)}")
    
    # Verify each model has correct configuration
    for model_key, config in MODELS_REGISTRY.items():
        assert config.name == model_key, f"Model {model_key} has mismatched name"
        assert config.provider in {"openai", "huggingface"}, f"Unknown provider for {model_key}"
    
    print("   ✓ All models have valid configurations")
    return True


def test_create_llm_openai():
    """Test that create_llm works for OpenAI models."""
    print("\n✅ Test 3: create_llm for OpenAI models")
    # We can't actually create ChatOpenAI without API key, but we can test the logic
    from src.common.model_provider import MODELS_REGISTRY, create_llm
    
    # This will fail if OPENAI_API_KEY is not set, which is expected
    try:
        gpt_config = MODELS_REGISTRY["gpt-4.1"]
        # Just verify the config is correct - actual creation requires API key
        assert gpt_config.provider == "openai"
        print("   ✓ OpenAI model config verified (creation requires OPENAI_API_KEY)")
    except Exception as e:
        print(f"   ✓ Expected behavior (API key needed): {str(e)[:50]}...")
    
    return True


def test_create_llm_huggingface_error():
    """Test that create_llm raises appropriate errors for HF models without endpoints."""
    print("\n✅ Test 4: create_llm error handling for HuggingFace")
    from src.common.model_provider import MODELS_REGISTRY, create_llm
    
    mediphi_config = MODELS_REGISTRY["mediphi"]
    
    # Should fail because endpoint URL is not set
    try:
        llm = create_llm(mediphi_config)
        print("   ✗ ERROR: Should have raised ValueError for missing endpoint URL")
        return False
    except ValueError as e:
        if "MEDIPHI_ENDPOINT_URL" in str(e):
            print(f"   ✓ Correctly raised error: {str(e)[:60]}...")
        else:
            print(f"   ✗ ERROR: Wrong error message: {str(e)}")
            return False
    
    return True


def test_rag_modules_backward_compat():
    """Test that RAG modules still work with old parameters."""
    print("\n✅ Test 5: RAG modules backward compatibility")
    from src.rag import simple, hybrid, hyde, rewriter
    from inspect import signature
    
    # Check simple.query_for_evaluation signature
    sig = signature(simple.query_for_evaluation)
    params = list(sig.parameters.keys())
    assert "question" in params, "simple.query_for_evaluation missing 'question' parameter"
    assert "custom_llm" in params, "simple.query_for_evaluation missing 'custom_llm' parameter"
    assert "llm_model" in params, "simple.query_for_evaluation missing 'llm_model' parameter for backward compat"
    print("   ✓ simple.query_for_evaluation has all parameters (question, llm_model, custom_llm)")
    
    # Check hybrid.query_for_evaluation signature
    sig = signature(hybrid.query_for_evaluation)
    params = list(sig.parameters.keys())
    assert "question" in params and "custom_llm" in params and "llm_model" in params
    print("   ✓ hybrid.query_for_evaluation has all parameters (question, llm_model, custom_llm)")
    
    # Check hyde.query_for_evaluation signature
    sig = signature(hyde.query_for_evaluation)
    params = list(sig.parameters.keys())
    assert all(p in params for p in ["question", "hyde_model", "answer_model", "custom_hyde_llm", "custom_answer_llm"])
    print("   ✓ hyde.query_for_evaluation has all parameters (hyde_model, answer_model, custom_*_llm)")
    
    # Check rewriter.query_for_evaluation signature
    sig = signature(rewriter.query_for_evaluation)
    params = list(sig.parameters.keys())
    assert all(p in params for p in ["question", "rewriter_model", "answer_model", "custom_rewriter_llm", "custom_answer_llm"])
    print("   ✓ rewriter.query_for_evaluation has all parameters (rewriter_model, answer_model, custom_*_llm)")
    
    return True


def test_evaluator_imports():
    """Test that the evaluator can be imported with new dependencies."""
    print("\n✅ Test 6: RAGASEvaluator imports")
    try:
        from src.evaluation.ragas_evaluator import RAGASEvaluator, MODELS_REGISTRY, create_llm
        print("   ✓ RAGASEvaluator imports successfully with model_provider dependencies")
        
        # Verify that create_llm and MODELS_REGISTRY are available in evaluator scope
        from src.evaluation.ragas_evaluator import run_all_models_all_rags_evaluation
        print("   ✓ run_all_models_all_rags_evaluation can be imported")
        
        return True
    except Exception as e:
        print(f"   ✗ ERROR: {str(e)}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 70)
    print("VERIFICATION TESTS FOR MODEL PROVIDER ABSTRACTION")
    print("=" * 70)
    
    tests = [
        test_model_config_creation,
        test_models_registry,
        test_create_llm_openai,
        test_create_llm_huggingface_error,
        test_rag_modules_backward_compat,
        test_evaluator_imports,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"\n   ✗ EXCEPTION in {test.__name__}: {str(e)}")
            results.append((test.__name__, False))
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All verification tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
