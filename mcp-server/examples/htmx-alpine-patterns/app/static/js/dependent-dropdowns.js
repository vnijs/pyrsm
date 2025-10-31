/**
 * Dependent Dropdowns Alpine.js Component
 *
 * Demonstrates:
 * - Reactive computed properties for exclusion logic
 * - State persistence with localStorage
 * - Progressive state restoration with $nextTick()
 */

function dependentDropdowns() {
    return {
        // State
        dataset: '',
        allColumns: [],
        xVariable: '',
        yVariable: '',

        // Computed: Available columns for X (exclude Y's selection)
        get availableForX() {
            if (!this.yVariable) {
                return this.allColumns;
            }
            return this.allColumns.filter(col => col !== this.yVariable);
        },

        // Computed: Available columns for Y (exclude X's selection)
        get availableForY() {
            if (!this.xVariable) {
                return this.allColumns;
            }
            return this.allColumns.filter(col => col !== this.xVariable);
        },

        // Initialize component
        init() {
            console.log('🚀 Dependent Dropdowns component initialized');
            this.restoreState();

            // If dataset was restored, load its columns
            if (this.dataset) {
                this.loadColumns(this.dataset);
            }
        },

        // Handle dataset change
        handleDatasetChange() {
            console.log(`📊 Dataset changed to: ${this.dataset}`);

            // Clear variable selections when dataset changes
            this.xVariable = '';
            this.yVariable = '';

            if (this.dataset) {
                this.loadColumns(this.dataset);
            } else {
                this.allColumns = [];
            }

            this.saveState();
        },

        // Load columns for selected dataset
        loadColumns(dataset) {
            console.log(`📥 Loading columns for dataset: ${dataset}`);

            // Simulate data (in real app, this would be an API call)
            const datasetColumns = {
                'salary': ['salary', 'rank', 'discipline', 'yrs_since_phd', 'yrs_service', 'sex'],
                'diamonds': ['price', 'carat', 'clarity', 'cut', 'color', 'depth', 'table', 'x', 'y', 'z'],
                'iris': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
            };

            this.allColumns = datasetColumns[dataset] || [];
            console.log(`✓ Loaded ${this.allColumns.length} columns:`, this.allColumns);
        },

        // Save state to localStorage
        saveState() {
            const state = {
                dataset: this.dataset,
                xVariable: this.xVariable,
                yVariable: this.yVariable,
                allColumns: this.allColumns
            };

            localStorage.setItem('dependentDropdowns', JSON.stringify(state));
            console.log('💾 State saved:', state);
        },

        // Restore state from localStorage
        restoreState() {
            const saved = localStorage.getItem('dependentDropdowns');

            if (!saved) {
                console.log('ℹ️ No saved state found');
                return;
            }

            try {
                const state = JSON.parse(saved);
                console.log('🔄 Restoring state:', state);

                // Step 1: Restore dataset and columns first
                this.dataset = state.dataset || '';
                this.allColumns = state.allColumns || [];

                // Step 2: Wait for DOM to update, then restore variables
                // This ensures dropdowns exist before we set their values
                this.$nextTick(() => {
                    // Restore X first
                    this.xVariable = state.xVariable || '';

                    // Then wait again and restore Y
                    // This ensures Y's available options are calculated after X is set
                    this.$nextTick(() => {
                        this.yVariable = state.yVariable || '';
                        console.log('✓ State fully restored');
                    });
                });

            } catch (error) {
                console.error('❌ Failed to restore state:', error);
            }
        },

        // Clear all state
        clearState() {
            console.log('🧹 Clearing all state');

            this.dataset = '';
            this.allColumns = [];
            this.xVariable = '';
            this.yVariable = '';

            localStorage.removeItem('dependentDropdowns');
            console.log('✓ State cleared');
        },

        // Simulate running analysis (for demonstration)
        simulateAnalysis() {
            console.log('🎯 Running analysis...');
            alert(`Would analyze: ${this.yVariable} ~ ${this.xVariable}\n\nDataset: ${this.dataset}`);
        }
    };
}

// Make globally available
window.dependentDropdowns = dependentDropdowns;
