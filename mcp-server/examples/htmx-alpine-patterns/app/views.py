"""
Views for HTMX + Alpine.js pattern examples
"""
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

# Sample data for examples
SAMPLE_DATASETS = {
    'salary': {
        'columns': ['salary', 'rank', 'discipline', 'yrs_since_phd', 'yrs_service', 'sex']
    },
    'diamonds': {
        'columns': ['price', 'carat', 'clarity', 'cut', 'color', 'depth', 'table', 'x', 'y', 'z']
    },
    'iris': {
        'columns': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    }
}

DISTRIBUTION_PARAMS = {
    'Normal': [
        {'name': 'mean', 'label': 'Mean', 'type': 'number', 'default': 0, 'step': 'any'},
        {'name': 'stdev', 'label': 'Standard deviation', 'type': 'number', 'default': 1, 'step': 'any', 'min': 0}
    ],
    'Binomial': [
        {'name': 'n', 'label': 'Number of trials', 'type': 'number', 'default': 10, 'step': 1, 'min': 1},
        {'name': 'p', 'label': 'Probability', 'type': 'number', 'default': 0.5, 'step': 0.01, 'min': 0, 'max': 1}
    ],
    'Poisson': [
        {'name': 'lambda', 'label': 'Rate (λ)', 'type': 'number', 'default': 1, 'step': 'any', 'min': 0}
    ],
    't': [
        {'name': 'df', 'label': 'Degrees of freedom', 'type': 'number', 'default': 1, 'step': 1, 'min': 1}
    ],
    'Uniform': [
        {'name': 'min', 'label': 'Minimum', 'type': 'number', 'default': 0, 'step': 'any'},
        {'name': 'max', 'label': 'Maximum', 'type': 'number', 'default': 1, 'step': 'any'}
    ]
}

TEST_PARAMETERS = {
    'ttest': [
        {'name': 'sample_type', 'label': 'Sample type', 'type': 'select',
         'options': [('independent', 'Independent'), ('paired', 'Paired')], 'default': 'independent'},
        {'name': 'confidence', 'label': 'Confidence level', 'type': 'number',
         'default': 0.95, 'step': 0.01, 'min': 0, 'max': 1}
    ],
    'wilcoxon': [
        {'name': 'sample_type', 'label': 'Sample type', 'type': 'select',
         'options': [('independent', 'Independent'), ('paired', 'Paired')], 'default': 'independent'}
    ],
    'chisquare': [
        {'name': 'correction', 'label': 'Yates correction', 'type': 'checkbox', 'default': False}
    ],
    'anova': [
        {'name': 'post_hoc', 'label': 'Post-hoc test', 'type': 'select',
         'options': [('none', 'None'), ('tukey', 'Tukey HSD'), ('bonferroni', 'Bonferroni')], 'default': 'none'}
    ]
}


def index(request):
    """Landing page with list of examples"""
    return render(request, 'patterns/index.html')


def dependent_dropdowns(request):
    """Example 1: Dependent dropdowns with variable exclusion"""
    context = {
        'datasets': list(SAMPLE_DATASETS.keys())
    }
    return render(request, 'patterns/dependent_dropdowns.html', context)


def conditional_visibility(request):
    """Example 2: Conditional visibility based on selections"""
    context = {
        'test_types': ['ttest', 'wilcoxon', 'chisquare', 'anova']
    }
    return render(request, 'patterns/conditional_visibility.html', context)


def state_restoration(request):
    """Example 3: Complex state restoration with UI dynamics"""
    context = {
        'distributions': list(DISTRIBUTION_PARAMS.keys()),
        'datasets': list(SAMPLE_DATASETS.keys())
    }
    return render(request, 'patterns/state_restoration.html', context)


@require_http_methods(["POST"])
def get_dataset_columns(request):
    """HTMX endpoint: Get columns for selected dataset"""
    dataset = request.POST.get('dataset')

    if dataset not in SAMPLE_DATASETS:
        return JsonResponse({'error': 'Dataset not found'}, status=404)

    columns = SAMPLE_DATASETS[dataset]['columns']

    # Return HTML for HTMX to swap
    html = ''.join([
        f'<option value="{col}">{col}</option>'
        for col in columns
    ])

    return render(request, 'patterns/partials/column_options.html', {
        'columns': columns
    })


@require_http_methods(["POST"])
def get_test_parameters(request):
    """HTMX endpoint: Get parameters for selected test type"""
    test_type = request.POST.get('test_type')

    if test_type not in TEST_PARAMETERS:
        return render(request, 'patterns/partials/test_params.html', {
            'params': []
        })

    params = TEST_PARAMETERS[test_type]

    return render(request, 'patterns/partials/test_params.html', {
        'params': params
    })


@require_http_methods(["POST"])
def get_distribution_params(request):
    """HTMX endpoint: Get parameters for selected distribution"""
    distribution = request.POST.get('distribution')

    if distribution not in DISTRIBUTION_PARAMS:
        return render(request, 'patterns/partials/distribution_params.html', {
            'fields': []
        })

    fields = DISTRIBUTION_PARAMS[distribution]

    return render(request, 'patterns/partials/distribution_params.html', {
        'fields': fields
    })
