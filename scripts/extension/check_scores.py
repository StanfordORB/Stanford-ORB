import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_method_scores(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def group_scenes_by_object(scenes):
    """Group scenes by their object name (after last underscore)"""
    object_groups = defaultdict(list)
    for scene in scenes:
        # Split by underscore and get the last part (object name)
        object_name = scene.split('_')[-1]
        object_groups[object_name].append(scene)
    return dict(sorted(object_groups.items()))  # Sort by object name

def analyze_scores():
    # Find all method JSON files in the directory
    json_dir = Path(__file__).parent / "from_meta/baselines"
    method_files = list(json_dir.glob("*.json"))
    
    if not method_files:
        print(f"No JSON files found in {json_dir}")
        return
        
    # Load all method data
    method_data = {}
    for file_path in method_files:
        method_name = file_path.stem
        method_data[method_name] = load_method_scores(file_path)
    
    # Get list of all scenes from first method
    first_method = next(iter(method_data.values()))
    scenes = list(first_method['scores']['view_all'].keys())
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent / "plots"
    output_dir.mkdir(exist_ok=True)
    
    # Create comparison plots for different test types
    test_types = {
        'View': {
            'source': 'view_all',
            'metrics': ['psnr_hdr', 'lpips', 'ssim']
        },
        'Light': {
            'source': 'light_all',
            'metrics': ['psnr_hdr', 'lpips', 'ssim']
        },
        'Geometry': {
            'source': 'geometry_all',
            'metrics': ['normal_angle', 'depth_mse']
        },
        'Shape': {
            'source': 'shape_all',
            'metrics': ['bidir_chamfer']
        }
    }
    
    # Group scenes by object
    object_groups = group_scenes_by_object(scenes)
    
    # Create one figure per test type
    for test_type, config in test_types.items():
        source = config['source']
        metrics = config['metrics']
        
        # Create subplot grid
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(15, 5*n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        for ax_idx, metric in enumerate(metrics):
            ax = axes[ax_idx]
            
            # Calculate average difficulty per object
            object_difficulties = {}
            for object_name, object_scenes in object_groups.items():
                scores = []
                for scene in object_scenes:
                    scene_scores = []
                    for data in method_data.values():
                        try:
                            score = data['scores'][source][scene][metric]
                            scene_scores.append(score)
                        except KeyError:
                            continue
                    if scene_scores:
                        scores.append(np.mean(scene_scores))
                if scores:
                    object_difficulties[object_name] = np.mean(scores)
            
            # Sort objects by difficulty
            reverse = metric not in ['lpips', 'normal_angle', 'depth_mse', 'bidir_chamfer']
            sorted_objects = sorted(object_difficulties.items(), key=lambda x: x[1], reverse=reverse)
            sorted_object_names = [obj[0] for obj in sorted_objects]
            
            # Sort scenes by object difficulty and then by scene difficulty
            sorted_scenes = []
            for object_name in sorted_object_names:
                object_scenes = object_groups[object_name]
                scene_scores = []
                for scene in object_scenes:
                    scores = []
                    for data in method_data.values():
                        try:
                            score = data['scores'][source][scene][metric]
                            scores.append(score)
                        except KeyError:
                            continue
                    if scores:  # Only include scenes that have valid scores
                        avg_score = np.mean(scores)
                        scene_scores.append((scene, avg_score))
                
                sorted_object_scenes = [s[0] for s in sorted(scene_scores, key=lambda x: x[1], reverse=reverse)]
                sorted_scenes.extend(sorted_object_scenes)
            
            # Plot with sorted scenes
            x = np.arange(len(sorted_scenes))
            width = 0.8 / len(method_data)
            
            for i, (method, data) in enumerate(method_data.items()):
                scores = []
                valid_x = []
                for idx, scene in enumerate(sorted_scenes):
                    try:
                        score = data['scores'][source][scene][metric]
                        if metric == 'depth_mse':
                            # Cap depth_mse outliers at 95th percentile
                            p95 = np.nanpercentile([s for s in scores if not np.isnan(s)], 95)
                            score = min(score, p95)
                        scores.append(score)
                        valid_x.append(idx)
                    except KeyError:
                        continue
                
                if scores:  # Only plot if we have valid scores
                    ax.bar(np.array(valid_x) + i*width, scores, width, 
                           label=f'{method}-{metric}', alpha=0.7)
            
            # Add vertical lines between object groups
            current_x = 0
            for object_name in sorted_object_names:
                object_scenes = [s for s in object_groups[object_name] if s in sorted_scenes]
                if not object_scenes:  # Skip empty object groups
                    continue
                current_x += len(object_scenes)
                if current_x < len(sorted_scenes):
                    ax.axvline(x=current_x - 0.1, color='black', linestyle='--', alpha=0.3)
                    avg_diff = object_difficulties[object_name]
                    ax.text(current_x - len(object_scenes)/2, ax.get_ylim()[1], 
                            f'{object_name}\n(avg: {avg_diff:.2f})', 
                            horizontalalignment='center', verticalalignment='bottom')
            
            ax.set_xlabel('Scenes')
            ax.set_ylabel('Scores')
            ax.set_title(f'{test_type} Metrics Comparison')
            ax.set_xticks(x + width*(len(method_data))/2)
            ax.set_xticklabels(sorted_scenes, rotation=45, ha='right')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        fig.tight_layout()
        output_path = output_dir / f"{test_type.lower()}_metrics.png"
        fig.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close(fig)  # Close the figure to free memory
        print(f"Saved plot to {output_path}")
        
        # Print statistics
        print(f"\n{test_type} Statistics:")
        print("-" * 70)
        print(f"{'Method-Metric':25} {'Mean':>10} {'Std':>10}")
        print("-" * 70)
        for method, data in method_data.items():
            for metric in metrics:
                scores = []
                for scene in scenes:
                    try:
                        score = data['scores'][source][scene][metric]
                        scores.append(score)
                    except KeyError:
                        continue
                if scores:
                    print(f"{method}-{metric:20} {np.mean(scores):10.3f} {np.std(scores):10.3f}")
    
    # Print final statistics
    print("\n" + "="*50)
    print("Final Statistics:")
    print(f"Total number of scenes: {len(scenes)}")
    print("\nScenes by object:")
    object_groups = group_scenes_by_object(scenes)
    for object_name, object_scenes in object_groups.items():
        print(f"\n{object_name}: {len(object_scenes)} scenes")
        # Sort scenes by their index number
        for scene in sorted(object_scenes, key=lambda x: int(x[5:8])):  # Extract index from "scene123_000_object"
            print(f"  - {scene}")
    print("="*50)

if __name__ == "__main__":
    analyze_scores()
