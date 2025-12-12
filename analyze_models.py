#!/usr/bin/env python3
"""
Analyze model classification and identify misclassified models.
"""
import json
import sys
from pathlib import Path

def load_registry():
    """Load model registry from cache"""
    cache_path = Path.home() / '.cache' / 'siliconflow' / 'model_registry.json'
    if not cache_path.exists():
        print(f"Registry not found: {cache_path}")
        sys.exit(1)
    
    with open(cache_path, 'r') as f:
        return json.load(f)

def analyze_chat_models(registry):
    """Analyze which models are marked as chat-capable"""
    chat_models = []
    non_chat_models = []
    
    # Check all categories
    categories = ['chat_models', 'vision_models', 'audio_models', 'video_models', 
                  'embedding_models', 'rerank_models']
    
    for category in categories:
        if category not in registry:
            continue
        
        for model in registry[category]['models']:
            model_id = model['id']
            supports_chat = model.get('supports_chat', False)
            
            # Check for problematic keywords in ID
            id_lower = model_id.lower()
            problematic_keywords = ['image', 'tts', 'speech', 'audio', 'video', 
                                   'embedding', 'rerank', 'vl', 'vision', 'visual']
            
            has_problematic = any(kw in id_lower for kw in problematic_keywords)
            
            info = {
                'id': model_id,
                'category': category,
                'supports_chat': supports_chat,
                'has_problematic_keyword': has_problematic,
                'keywords_found': [kw for kw in problematic_keywords if kw in id_lower]
            }
            
            if supports_chat:
                chat_models.append(info)
            else:
                non_chat_models.append(info)
    
    return chat_models, non_chat_models

def main():
    registry = load_registry()
    
    print("📊 Model Analysis Report")
    print("=" * 80)
    
    chat_models, non_chat_models = analyze_chat_models(registry)
    
    print(f"\n✅ Chat-capable models ({len(chat_models)}):")
    for model in sorted(chat_models, key=lambda x: x['id']):
        if model['has_problematic_keyword']:
            print(f"  ⚠️  {model['id']} (category: {model['category']}, keywords: {model['keywords_found']})")
        else:
            print(f"  ✓ {model['id']}")
    
    print(f"\n❌ Non-chat models ({len(non_chat_models)}):")
    for model in sorted(non_chat_models, key=lambda x: x['id']):
        print(f"  - {model['id']} (category: {model['category']})")
    
    # Identify potentially misclassified models
    print("\n🔍 Potentially misclassified models (chat-capable but have problematic keywords):")
    problematic = [m for m in chat_models if m['has_problematic_keyword']]
    for model in problematic:
        print(f"  ❓ {model['id']}")
        print(f"     Category: {model['category']}")
        print(f"     Keywords: {model['keywords_found']}")
    
    # Summary
    print("\n📈 Summary:")
    print(f"  Total chat-capable models: {len(chat_models)}")
    print(f"  Total non-chat models: {len(non_chat_models)}")
    print(f"  Potentially misclassified: {len(problematic)}")
    
    if problematic:
        print("\n💡 Recommendations:")
        print("  1. Add missing keywords to specialized_keywords list in siliconflow_client.py")
        print("  2. Check model types in API response")
        print("  3. Review capability inference logic")

if __name__ == '__main__':
    main()