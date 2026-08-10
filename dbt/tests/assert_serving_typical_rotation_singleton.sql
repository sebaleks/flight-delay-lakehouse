-- serving_typical_rotation is contractually EXACTLY ONE ROW.
--
-- ml/serving.py pins every context-less prediction — i.e. every consumer
-- request — to that row. If the model ever produced more than one, serving
-- would take whichever BigQuery returned first, making the typical rotation
-- profile depend on scan order across deploys. That is precisely the class of
-- nondeterminism the exact-median change removed (approx_quantiles was
-- observed returning four different values for the same median), so it gets a
-- guard rather than a comment.
--
-- Fails with a row count for any deviation, including zero rows: an empty
-- table would send serving down the RuntimeError path at startup, which is
-- correct behaviour but should be caught here at build time instead.
--
-- The plausible regressions this catches: a grouping key added to the
-- `attributes` CTE, or a NULL/NaN defeating its `select distinct` dedup
-- (NaN != NaN, so two NaN rows survive DISTINCT).

select count(*) as n_rows
from {{ ref('serving_typical_rotation') }}
having count(*) != 1
