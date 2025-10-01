#!/usr/bin/env python3
"""
Analysis Script for IEEE EMBS BHI 2025 Conference Submission
Run this after main_experiment.py to generate Conference_Submission and Figures
"""

import sys
import os
from pathlib import Path
import traceback

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Generate Conference_Submission folder and all figures from existing results."""
    
    print("=" * 60)
    print("🎯 IEEE EMBS BHI 2025 - CONFERENCE SUBMISSION GENERATOR")
    print("=" * 60)
    
    # Check if Results_12W folder exists
    results_folder = "../Results_24W"
    if not Path(results_folder).exists():
        print(f"❌ ERROR: Results folder not found: {results_folder}")
        print("   Please run main_experiment.py first!")
        return False
    
    print(f"✅ Found results folder: {results_folder}")
    
    try:
        # Import and initialize ResultsCompilation
        print("\n📋 Initializing Results Compilation...")
        from results_compilation import ResultsCompilation
        
        # Initialize with auto-loading
        results_compiler = ResultsCompilation(results_folder)
        
        # Check if results were loaded
        if not results_compiler.all_results:
            print("❌ ERROR: No results were auto-loaded!")
            return False
        
        print(f"✅ Auto-loaded {len(results_compiler.all_results)} phases")
        total_models = sum(len(phase_results) for phase_results in results_compiler.all_results.values())
        print(f"✅ Total models: {total_models}")
        
        # Create Conference_Submission folder
        print("\n📁 Creating Conference_Submission folder...")
        conf_dir = Path(results_folder) / "Conference_Submission"
        conf_dir.mkdir(exist_ok=True)
        print(f"✅ Conference_Submission folder ready: {conf_dir}")
        
        # Generate all tables
        print("\n📊 Creating Publication Tables...")
        try:
            perf_table = results_compiler.create_performance_summary_table()
            phase_table = results_compiler.create_phase_comparison_table()
            stat_table = results_compiler.create_statistical_significance_table()
            clinical_table = results_compiler.create_clinical_significance_table()
            print("✅ All tables created!")
        except Exception as e:
            print(f"⚠️ Tables creation had issues: {e}")
        
        # Generate all figures
        print("\n🎨 Creating All 14 Publication Figures...")
        try:
            results_compiler.create_publication_figures()
            print("✅ All 14 figures created!")
        except Exception as e:
            print(f"❌ Figures creation failed: {e}")
            traceback.print_exc()
            return False
        
        # Generate conference summary
        print("\n📝 Generating Conference Summary...")
        try:
            summary = results_compiler.generate_conference_summary()
            print("✅ Conference summary generated!")
        except Exception as e:
            print(f"⚠️ Summary generation had issues: {e}")
        
        # Save all results to Conference_Submission
        print("\n💾 Saving All Results to Conference_Submission...")
        try:
            results_compiler.save_all_results()
            print("✅ All results saved!")
        except Exception as e:
            print(f"❌ Saving failed: {e}")
            traceback.print_exc()
            return False
        
        # Verify what was created
        print("\n🔍 Verification:")
        if conf_dir.exists():
            contents = list(conf_dir.glob("*"))
            print(f"✅ Conference_Submission contains {len(contents)} items:")
            for item in contents:
                print(f"   📄 {item.name}")
        
        figures_dir = Path(results_folder) / "Figures"
        if figures_dir.exists():
            figures = list(figures_dir.glob("*.png"))
            print(f"✅ Figures folder contains {len(figures)} PNG files:")
            for fig in sorted(figures):
                print(f"   🖼️ {fig.name}")
        
        print("\n" + "=" * 60)
        print("🎉 CONFERENCE SUBMISSION GENERATION COMPLETE!")
        print("=" * 60)
        print(f"📁 Conference_Submission: {conf_dir}")
        print(f"🖼️ Figures: {figures_dir}")
        print("🚀 Ready for IEEE EMBS BHI 2025 submission!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Analysis completed successfully!")
        else:
            print("\n❌ Analysis failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)