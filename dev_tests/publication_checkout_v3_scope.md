# Exact checkout-only source succession

```diff
diff --git a/.gitattributes b/.gitattributes
index 1933679..e40ec05 100644
--- a/.gitattributes
+++ b/.gitattributes
@@ -14,3 +14,7 @@
 # Byte-exact historical fixtures must survive checkout without conversion.
 /dev_tests/fixtures/** -text -whitespace
 /docs/README_TENSORFLOW_LEGACY.md -whitespace
+/tutorial/README.md text eol=lf
+/dev_tests/adabelief_moment_scope.diff -text -whitespace
+/dev_tests/publication_v1_scope.diff -text -whitespace
+/dev_tests/publication_windows_v2_scope.diff -text -whitespace
diff --git a/dev_tests/adabelief_moment_scope.py b/dev_tests/adabelief_moment_scope.py
index a961730..d2c152f 100644
--- a/dev_tests/adabelief_moment_scope.py
+++ b/dev_tests/adabelief_moment_scope.py
@@ -94,7 +94,11 @@ def assert_adabelief_phase_scope_v4():
         if name.startswith('dev_tests/artifacts/'):
             continue
         if name not in TRANSITIONS and name not in portable and (name.startswith(('nif/','tutorial/','dev_tests/','tests/')) or name=='.gitattributes'):
-            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
+            if name=='.gitattributes':
+                from dev_tests.publication_windows_scope import assert_checkout_attributes
+                assert_checkout_attributes(ROOT,digest)
+            else:
+                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
     # Freeze the new oracle at its BEFORE-fix hash. The native source entry in
     # that RED manifest identifies the original snapshot, not the new candidate.
     red=json.loads((ROOT/'dev_tests/adabelief_moment_red_fingerprints.json').read_text(encoding='utf-8'))
diff --git a/dev_tests/publication_windows_scope.py b/dev_tests/publication_windows_scope.py
index df43dfd..64df3ed 100644
--- a/dev_tests/publication_windows_scope.py
+++ b/dev_tests/publication_windows_scope.py
@@ -71,6 +71,8 @@ def expected_source(name,source):
     for old,new in TRANSITIONS[name]:
         assert source.count(old)==1,(name,old)
         source=source.replace(old,new,1)
+    if name=='dev_tests/adabelief_moment_scope.py':
+        source=expected_checkout_source(source)
     return source
 
 
@@ -93,3 +95,26 @@ def assert_publication_windows_scope_v2(root=ROOT):
             assert archive[name]['sha256']==digest,name
         else:assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
     return set(TRANSITIONS)
+
+
+def expected_checkout_source(source):
+    """Checkout v3 permits only the exact attributes guard succession."""
+    old='            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name\n'
+    new=("            if name=='.gitattributes':\n"
+         '                from dev_tests.publication_windows_scope import assert_checkout_attributes\n'
+         '                assert_checkout_attributes(ROOT,digest)\n'
+         '            else:\n'
+         '                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name\n')
+    assert source.count(old)==1
+    return source.replace(old,new,1)
+
+
+def assert_checkout_attributes(root,digest):
+    """Keep the historical bytes/hash; allow only four exact checkout rules."""
+    suffix=(b'/tutorial/README.md text eol=lf\n'
+            b'/dev_tests/adabelief_moment_scope.diff -text -whitespace\n'
+            b'/dev_tests/publication_v1_scope.diff -text -whitespace\n'
+            b'/dev_tests/publication_windows_v2_scope.diff -text -whitespace\n')
+    current=(Path(root)/'.gitattributes').read_bytes()
+    assert current.endswith(suffix),'.gitattributes'
+    assert hashlib.sha256(current[:-len(suffix)]).hexdigest()==digest,'.gitattributes'
```
