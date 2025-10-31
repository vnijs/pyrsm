/**
 * Conditional Visibility Alpine.js Component
 *
 * Demonstrates:
 * - Dynamic visibility with x-show
 * - HTMX partial updates for server-rendered forms
 * - State restoration after HTMX swaps
 * - Integration between Alpine reactivity and HTMX
 */

function conditionalVisibility() {
    return {
        // State
        testType: '',
        savedParams: {},  // Store parameters for each test type

        // Computed: Should conditional params be visible?
        get showConditionalParams() {
            // Some tests have conditional parameters, some don't
            return ['ttest', 'wilcoxon', 'anova'].includes(this.testType);
        },

        // Initialize
        init() {
            console.log('🚀 Conditional Visibility component initialized');
            this.restoreState();

            // If test type was restored, trigger HTMX load
            if (this.testType) {
                this.$nextTick(() => {
                    this.loadTestParameters();
                });
            }
        },

        // Handle test type change
        handleTestChange() {
            console.log(`🧪 Test type changed to: ${this.testType}`);

            if (this.testType) {
                this.loadTestParameters();
            }

            this.saveState();
        },

        // Load parameters for current test type via HTMX
        loadTestParameters() {
            console.log(`📥 Loading parameters for: ${this.testType}`);

            const container = document.getElementById('test-params-container');
            if (!container) {
                console.warn('⚠️ Container not found');
                return;
            }

            // Create FormData and trigger HTMX
            const formData = new FormData();
            formData.append('test_type', this.testType);

            // Get CSRF token
            const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value
                || document.cookie.match(/csrftoken=([^;]+)/)?.[1];

            if (csrfToken) {
                formData.append('csrfmiddlewaretoken', csrfToken);
            }

            // Manual HTMX request
            htmx.ajax('POST', container.getAttribute('hx-post'), {
                target: '#test-params-container',
                swap: 'innerHTML',
                values: {
                    test_type: this.testType
                }
            });
        },

        // Restore parameter values after HTMX swaps new HTML
        restoreParamValues() {
            console.log('🔄 Restoring parameter values after HTMX swap');

            if (!this.testType || !this.savedParams[this.testType]) {
                console.log('ℹ️ No saved params for this test type');
                return;
            }

            const params = this.savedParams[this.testType];
            console.log('📦 Restoring params:', params);

            // Wait for DOM to settle
            this.$nextTick(() => {
                Object.entries(params).forEach(([name, value]) => {
                    const input = document.querySelector(`[name="${name}"]`);
                    if (input) {
                        if (input.type === 'checkbox') {
                            input.checked = value;
                        } else {
                            input.value = value;
                        }
                        console.log(`  ✓ Restored ${name} = ${value}`);
                    }
                });
            });
        },

        // Save current parameter values before HTMX swap
        saveCurrentParams() {
            if (!this.testType) return;

            console.log(`💾 Saving params for: ${this.testType}`);

            const container = document.getElementById('test-params-container');
            if (!container) return;

            const params = {};
            const inputs = container.querySelectorAll('input, select, textarea');

            inputs.forEach(input => {
                if (input.name) {
                    params[input.name] = input.type === 'checkbox' ? input.checked : input.value;
                }
            });

            this.savedParams[this.testType] = params;
            console.log('  Saved:', params);
            this.saveState();
        },

        // Save state to localStorage
        saveState() {
            const state = {
                testType: this.testType,
                savedParams: this.savedParams
            };

            localStorage.setItem('conditionalVisibility', JSON.stringify(state));
            console.log('💾 State saved:', state);
        },

        // Restore state from localStorage
        restoreState() {
            const saved = localStorage.getItem('conditionalVisibility');

            if (!saved) {
                console.log('ℹ️ No saved state');
                return;
            }

            try {
                const state = JSON.parse(saved);
                console.log('🔄 Restoring state:', state);

                this.testType = state.testType || '';
                this.savedParams = state.savedParams || {};

                console.log('✓ State restored');
            } catch (error) {
                console.error('❌ Failed to restore state:', error);
            }
        },

        // Clear all state
        clearState() {
            console.log('🧹 Clearing state');

            this.testType = '';
            this.savedParams = {};

            // Clear the parameters container
            const container = document.getElementById('test-params-container');
            if (container) {
                container.innerHTML = '<p class="text-muted">Select a test type to see parameters</p>';
            }

            localStorage.removeItem('conditionalVisibility');
            console.log('✓ Cleared');
        },

        // Simulate running test
        runTest() {
            // Capture current parameter values
            this.saveCurrentParams();

            const params = this.savedParams[this.testType] || {};
            const paramsStr = Object.entries(params)
                .map(([k, v]) => `${k}: ${v}`)
                .join('\n');

            alert(`Would run ${this.testType} test\n\nParameters:\n${paramsStr || 'None'}`);
        }
    };
}

// Setup HTMX event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Before HTMX swaps, save current parameter values
    document.body.addEventListener('htmx:beforeSwap', (event) => {
        if (event.detail.target.id === 'test-params-container') {
            // Get Alpine component
            const el = document.querySelector('[x-data*="conditionalVisibility"]');
            if (el && el.__x) {
                el.__x.$data.saveCurrentParams();
            }
        }
    });
});

window.conditionalVisibility = conditionalVisibility;
