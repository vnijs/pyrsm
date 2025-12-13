"""Tests for pyrsm.eda module."""

import pytest
import polars as pl
import numpy as np
from pyrsm.eda import explore, pivot, combine, visualize


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    return pl.DataFrame({
        "price": np.random.uniform(100, 1000, 100).tolist(),
        "quantity": np.random.randint(1, 20, 100).tolist(),
        "category": np.random.choice(["A", "B", "C"], 100).tolist(),
        "region": np.random.choice(["North", "South"], 100).tolist(),
    })


@pytest.fixture
def join_data():
    """Create data for join testing."""
    orders = pl.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "customer_id": [101, 102, 101, 103, 104],
        "amount": [100.0, 200.0, 150.0, 300.0, 250.0],
    })
    customers = pl.DataFrame({
        "customer_id": [101, 102, 103, 105],
        "name": ["Alice", "Bob", "Charlie", "Eve"],
    })
    return orders, customers


class TestExplore:
    """Tests for explore function."""

    def test_explore_default(self, sample_data):
        """Test explore with default settings."""
        result = explore(sample_data)
        assert isinstance(result, pl.DataFrame)
        # Without by: stats as rows, variables as columns
        assert "statistic" in result.columns
        assert "price" in result.columns
        assert "quantity" in result.columns
        assert "mean" in result["statistic"].to_list()

    def test_explore_specific_columns(self, sample_data):
        """Test explore with specific columns."""
        result = explore(sample_data, cols=["price"])
        assert "price" in result.columns
        assert "quantity" not in result.columns

    def test_explore_custom_functions(self, sample_data):
        """Test explore with custom functions."""
        result = explore(sample_data, cols=["price"], agg=["mean", "median", "min", "max"])
        assert "price" in result.columns
        stats = result["statistic"].to_list()
        assert "mean" in stats
        assert "median" in stats
        assert "min" in stats
        assert "max" in stats
        assert "std" not in stats

    def test_explore_grouped(self, sample_data):
        """Test explore with grouping."""
        result = explore(sample_data, cols=["price"], by="category")
        assert "category" in result.columns
        assert len(result) == 3  # A, B, C

    def test_explore_invalid_function(self, sample_data):
        """Test explore with invalid function raises error."""
        with pytest.raises(ValueError, match="Unknown aggregation function"):
            explore(sample_data, agg=["invalid_func"])

    def test_explore_lazyframe(self, sample_data):
        """Test explore with LazyFrame input."""
        result = explore(sample_data.lazy())
        assert isinstance(result, pl.DataFrame)

    def test_explore_all_functions(self, sample_data):
        """Test all supported functions."""
        agg = ["mean", "median", "sum", "std", "var", "min", "max", "count", "n_unique", "null_count"]
        result = explore(sample_data, cols=["price"], agg=agg)
        stats = result["statistic"].to_list()
        for func in agg:
            assert func in stats


class TestPivot:
    """Tests for pivot function."""

    def test_pivot_frequency_table(self, sample_data):
        """Test single variable frequency table."""
        result = pivot(sample_data, rows="category")
        assert "category" in result.columns
        assert "count" in result.columns
        assert len(result) == 3  # A, B, C

    def test_pivot_crosstab(self, sample_data):
        """Test two-variable crosstab."""
        result = pivot(sample_data, rows="category", cols="region")
        assert "category" in result.columns
        assert "North" in result.columns or "South" in result.columns

    def test_pivot_with_values(self, sample_data):
        """Test pivot with value aggregation."""
        result = pivot(sample_data, rows="category", values="price", agg="mean")
        assert "category" in result.columns
        assert "price_mean" in result.columns

    def test_pivot_with_totals(self, sample_data):
        """Test pivot with totals."""
        result = pivot(sample_data, rows="category", totals=True)
        # Last row should be "Total"
        assert result["category"].to_list()[-1] == "Total"

    def test_pivot_normalize_row(self, sample_data):
        """Test pivot with row normalization."""
        result = pivot(sample_data, rows="category", cols="region", normalize="row", totals=True)
        # Row totals should be 100%
        total_col = result.filter(pl.col("category") != "Total")["Total"]
        assert all(abs(v - 100.0) < 0.01 for v in total_col.to_list())

    def test_pivot_invalid_agg(self, sample_data):
        """Test pivot with invalid aggregation raises error."""
        with pytest.raises(ValueError, match="Unknown aggregation"):
            pivot(sample_data, rows="category", agg="invalid")

    def test_pivot_lazyframe(self, sample_data):
        """Test pivot with LazyFrame input."""
        result = pivot(sample_data.lazy(), rows="category")
        assert isinstance(result, pl.DataFrame)


class TestCombine:
    """Tests for combine function."""

    def test_inner_join(self, join_data):
        """Test inner join."""
        orders, customers = join_data
        result = combine(orders, customers, by="customer_id", type="inner_join")
        assert "name" in result.columns
        # Only customers 101, 102, 103 are in both
        assert len(result) == 4  # orders 1, 2, 3, 4 have matching customers

    def test_left_join(self, join_data):
        """Test left join."""
        orders, customers = join_data
        result = combine(orders, customers, by="customer_id", type="left_join")
        assert len(result) == 5  # All orders kept

    def test_right_join(self, join_data):
        """Test right join."""
        orders, customers = join_data
        result = combine(orders, customers, by="customer_id", type="right_join")
        # All customers kept, including Eve (105) with no orders
        customer_ids = set(result["customer_id"].to_list())
        assert 105 in customer_ids  # Eve should be included

    def test_bind_rows(self, sample_data):
        """Test bind_rows."""
        df1 = sample_data.head(50)
        df2 = sample_data.tail(50)
        result = combine(df1, df2, type="bind_rows")
        assert len(result) == 100

    def test_bind_cols(self):
        """Test bind_cols."""
        df1 = pl.DataFrame({"a": [1, 2, 3]})
        df2 = pl.DataFrame({"b": [4, 5, 6]})
        result = combine(df1, df2, type="bind_cols")
        assert "a" in result.columns
        assert "b" in result.columns

    def test_intersect(self):
        """Test intersect."""
        df1 = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        df2 = pl.DataFrame({"x": [2, 3, 4], "y": ["b", "c", "d"]})
        result = combine(df1, df2, type="intersect")
        assert len(result) == 2  # rows (2, "b") and (3, "c")

    def test_union(self):
        """Test union."""
        df1 = pl.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        df2 = pl.DataFrame({"x": [2, 3], "y": ["b", "c"]})
        result = combine(df1, df2, type="union")
        assert len(result) == 3  # Unique rows

    def test_setdiff(self):
        """Test setdiff."""
        df1 = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        df2 = pl.DataFrame({"x": [2, 3], "y": ["b", "c"]})
        result = combine(df1, df2, type="setdiff")
        assert len(result) == 1  # Only (1, "a")

    def test_join_requires_by(self, join_data):
        """Test that join types require by parameter."""
        orders, customers = join_data
        with pytest.raises(ValueError, match="requires by="):
            combine(orders, customers, type="inner_join")

    def test_invalid_type(self, join_data):
        """Test invalid combine type raises error."""
        orders, customers = join_data
        with pytest.raises(ValueError, match="Unknown combine type"):
            combine(orders, customers, type="invalid_type")

    def test_different_key_names(self, join_data):
        """Test join with different key names."""
        orders, customers = join_data
        # Rename customer_id in customers to cust_id
        customers_renamed = customers.rename({"customer_id": "cust_id"})
        result = combine(orders, customers_renamed, by="customer_id:cust_id", type="inner_join")
        assert "name" in result.columns


class TestVisualize:
    """Tests for visualize function."""

    def test_visualize_histogram(self, sample_data):
        """Test histogram creation."""
        p = visualize(sample_data, x="price")
        assert p is not None

    def test_visualize_scatter(self, sample_data):
        """Test scatter plot creation."""
        p = visualize(sample_data, x="price", y="quantity")
        assert p is not None

    def test_visualize_bar(self, sample_data):
        """Test bar chart for categorical."""
        p = visualize(sample_data, x="category", geom="bar")
        assert p is not None

    def test_visualize_box(self, sample_data):
        """Test box plot."""
        p = visualize(sample_data, x="category", y="price", geom="box")
        assert p is not None

    def test_visualize_with_color(self, sample_data):
        """Test plot with color aesthetic."""
        p = visualize(sample_data, x="price", y="quantity", color="category")
        assert p is not None

    def test_visualize_with_facet(self, sample_data):
        """Test faceted plot."""
        p = visualize(sample_data, x="price", facet="category")
        assert p is not None

    def test_visualize_smooth(self, sample_data):
        """Test scatter with smooth line."""
        p = visualize(sample_data, x="price", y="quantity", smooth="lm")
        assert p is not None

    def test_visualize_invalid_geom(self, sample_data):
        """Test invalid geom raises error."""
        with pytest.raises(ValueError, match="Unknown geom"):
            visualize(sample_data, x="price", geom="invalid")

    def test_visualize_missing_y(self, sample_data):
        """Test geom requiring y raises error when y missing."""
        with pytest.raises(ValueError, match="y is required"):
            visualize(sample_data, x="price", geom="scatter")

    def test_visualize_lazyframe(self, sample_data):
        """Test visualize with LazyFrame input."""
        p = visualize(sample_data.lazy(), x="price")
        assert p is not None

    def test_visualize_density(self, sample_data):
        """Test density plot."""
        p = visualize(sample_data, x="price", geom="density")
        assert p is not None

    def test_visualize_violin(self, sample_data):
        """Test violin plot."""
        p = visualize(sample_data, x="category", y="price", geom="violin")
        assert p is not None
