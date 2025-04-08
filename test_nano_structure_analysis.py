import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import persim
from image_feature_embedding import compute_persistence_homology, visualize_persistence_diagrams, multi_scale_filtration, analyze_nano_structure

def generate_sample_nano_structure(size=128, num_particles=50, min_radius=3, max_radius=10):
    """ナノ粒子構造のサンプル画像を生成する"""
    img_array = np.zeros((size, size))
    
    for _ in range(num_particles):
        x, y = np.random.randint(0, size, 2)
        r = np.random.randint(min_radius, max_radius)
        for i in range(-r, r):
            for j in range(-r, r):
                if 0 <= x+i < size and 0 <= y+j < size and i*i + j*j <= r*r:
                    img_array[x+i, y+j] = 1
    
    return img_array

if __name__ == "__main__":
    print("ナノ構造画像のパーシステントホモロジー解析テスト")
    print("=" * 50)
    
    print("\n1. サンプルナノ構造画像の生成")
    img_array = generate_sample_nano_structure()
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    
    plt.figure(figsize=(5, 5))
    plt.imshow(img_array, cmap='gray')
    plt.title('Sample Nano-structure Image')
    plt.savefig('sample_nano_structure.png')
    plt.show()
    
    print("\n2. 標準的なパーシステントホモロジー解析")
    dgms = compute_persistence_homology(img)
    fig = visualize_persistence_diagrams(dgms, title='Persistence Diagram for Sample Nano-structure')
    plt.savefig('persistence_diagram.png')
    
    print("\n3. 多スケールパーシステントホモロジー解析（革新的アプローチ）")
    filtrations = multi_scale_filtration(img)
    
    for i, filtration in enumerate(filtrations):
        scale = filtration['scale']
        print(f"スケール {scale} での解析結果:")
        diagram = filtration['diagram']
        
        plt.figure(figsize=(5, 5))
        persim.plot_diagrams(diagram)
        plt.title(f'Multi-scale Persistence Diagram (Scale {scale})')
        plt.savefig(f'multi_scale_diagram_{i}.png')
        plt.show()
    
    print("\n4. 解析結果のまとめ")
    images = {'sample_nano_structure': img}
    
    methods = ['persistence', 'multi_scale', 'combined']
    for method in methods:
        print(f"\n{method}手法による解析:")
        results = analyze_nano_structure(images, method=method)
        
        if method == 'persistence':
            for filename, result in results.items():
                features = result['features']
                print(f"ファイル '{filename}' の特徴量:")
                for dim, feature in enumerate(features):
                    print(f"  次元 {dim}: {feature}")
        
        elif method == 'multi_scale':
            for filename, result in results.items():
                multi_scale_features = result['multi_scale_features']
                print(f"ファイル '{filename}' の多スケール特徴量:")
                for feature in multi_scale_features:
                    print(f"  スケール {feature['scale']}")
        
        elif method == 'combined':
            for filename, feature in results.items():
                print(f"ファイル '{filename}' の組み合わせ特徴量: 形状 {feature.shape}, ノルム {np.linalg.norm(feature):.4f}")
    
    print("\n解析完了！")
