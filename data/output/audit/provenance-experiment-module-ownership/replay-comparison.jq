# Compare every retained leaf; execution identity is reported separately.
def magnitude: if . < 0 then -. else . end;
def scientific($name):
  if $name == "dataset.json" then del(.identity, .provenance.sources, .provenance.executable_sha256)
  elif $name == "summary.json" then del(.identity)
  else {kind, index, payload} end;
def category($path):
  if ($path | index("coefficients")) != null then "coefficients"
  elif ($path | index("penalized_deviance_trajectory")) != null then "deviance_trajectory"
  elif ($path | index("training_mean")) != null then "training_means"
  elif ($path | index("training_population_std")) != null then "training_population_std"
  elif ($path | index("log_loss")) != null then "log_loss"
  elif ($path | index("roc_auc")) != null then "roc_auc"
  elif ($path | index("average_precision")) != null then "average_precision"
  else ($path[-1] | tostring) end;
def compare($name; $left; $right):
  ($left | scientific($name)) as $original |
  ($right | scientific($name)) as $replay |
  ([$original | paths] | sort) as $original_paths |
  ([$replay | paths] | sort) as $replay_paths |
  [($original_paths + $replay_paths | unique)[] as $path |
    ($original | getpath($path)) as $before |
    ($replay | getpath($path)) as $after |
    select(($before | type) != ($after | type)) |
    {path:$path, original_type:($before|type), replay_type:($after|type)}] as $type_changes |
  [$original | paths(scalars) as $path |
    getpath($path) as $before | ($replay|getpath($path)) as $after |
    select($before != $after) |
    {path:$path, original:$before, replay:$after} |
    if ($before|type) == "number" and ($after|type) == "number" then
      . + {category:category($path), absolute_difference:(($after-$before)|magnitude),
           scale:([1, ($before|magnitude), ($after|magnitude)]|max)} |
      . + {normalized_difference:(.absolute_difference/.scale)}
    else . end] as $changes |
  {name:$name, original_identity:$left.identity, replay_identity:$right.identity,
   original_payload_sha256:$left.payload_sha256, replay_payload_sha256:$right.payload_sha256,
   scientific_exact:($original == $replay), path_sets_exact:($original_paths == $replay_paths),
   type_changes:$type_changes,
   nonnumeric_changes:[$changes[]|select(has("absolute_difference")|not)],
   integer_valued_original_changes:[$changes[]|select(has("absolute_difference"))|select(.original == (.original|floor))],
   numerical_changes:[$changes[]|select(has("absolute_difference"))],
   numerical_categories:([$changes[]|select(has("absolute_difference"))]|group_by(.category)|map({category:.[0].category,changed_leaves:length,maximum_absolute_difference:(map(.absolute_difference)|max),maximum_normalized_difference:(map(.normalized_difference)|max)}))};

[inputs | {file:input_filename, document:.}] as $inputs |
[$inputs[] | select(.file|startswith($original_root + "/")) |
  {key:(.file|ltrimstr($original_root + "/")), value:.document}] | from_entries as $originals |
[$inputs[] | select(.file|startswith($replay_root + "/")) |
  {key:(.file|ltrimstr($replay_root + "/")), value:.document}] | from_entries as $replays |
($originals|keys) as $expected |
($replays|keys) as $observed |
[$observed[] as $name | select($originals|has($name)) |
  compare($name; $originals[$name]; $replays[$name])] as $records |
{schema_version:1,
 comparison_boundary:"JSON path/type and nonnumeric equality; every changed numeric leaf retained. Scale is max(1, absolute original, absolute replay). Numerical differences are measured without a posthoc acceptance tolerance.",
 excluded_execution_fields:["identity", "payload_sha256", "dataset.provenance.sources", "dataset.provenance.executable_sha256"],
 expected_files:$expected, compared_files:$observed,
 missing_files:($expected-$observed), unexpected_files:($observed-$expected),
 complete_file_set:($expected==$observed),
 all_path_sets_exact:($records|all(.path_sets_exact)),
 all_types_exact:($records|all(.type_changes|length==0)),
 all_nonnumeric_leaves_exact:($records|all(.nonnumeric_changes|length==0)),
 all_integer_valued_original_leaves_exact:($records|all(.integer_valued_original_changes|length==0)),
 numerical_changed_leaves:([$records[].numerical_changes[]]|length),
 numerical_categories:([$records[].numerical_changes[]]|group_by(.category)|map({category:.[0].category,changed_leaves:length,maximum_absolute_difference:(map(.absolute_difference)|max),maximum_normalized_difference:(map(.normalized_difference)|max)})),
 records:$records}
