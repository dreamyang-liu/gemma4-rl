# The 106 local dev tasks (same distribution as the hidden Kaggle tasks)

## fastapi_15661 (fastapi/fastapi) gold: 1 file(s), 216 changed lines: scripts/prepare_release.py; tests: tests/test_prepare_release.py
👷 Automate release preparation

## Pull Request

👷 Automate release preparation

<!--
Please start with a GitHub Discussion.

Once a team member asks you to open a PR, create it and link the discussion here.

Obvious typo fixes can be made in a PR without starting a discussion.
-->

Discussion: <!-- Link to the GitHub Discussion -->

## Description

<!-- Write the description of your PR here -->

## AI Disclaimer

Codex with GPT-5.5, based on https://github.com/fastapi/asyncer/pull/608, reviewed by hand.

<!-- If using AI, write here the prompt and model used -->

<details>
<summary>AI transcr
...[387 more chars]

## fastapi_15588 (fastapi/fastapi) gold: 1 file(s), 23 changed lines: fastapi/sse.py; tests: tests/test_sse.py
♻️ Validate Server Sent Event fields to avoid applications from sending broken data

## Pull Request

<!--
Please start with a GitHub Discussion.

Once a team member asks you to open a PR, create it and link the discussion here.

Obvious typo fixes can be made in a PR without starting a discussion.
-->

Discussion: <!-- Link to the GitHub Discussion -->

## Description

♻️ Validate Server Sent Event fields to avoid applications from sending broken data

This was reported multiple times as a "vulnerability" by users probably using automated AI tools, with the argument that if input was taken fr
...[1247 more chars]

## fastapi_15589 (fastapi/fastapi) gold: 1 file(s), 4 changed lines: fastapi/dependencies/utils.py; tests: tests/test_query_cookie_header_model_extra_params.py
♻️ Do not accept underscore headers when using `convert_underscores=True` (the default)

## Pull Request

<!--
Please start with a GitHub Discussion.

Once a team member asks you to open a PR, create it and link the discussion here.

Obvious typo fixes can be made in a PR without starting a discussion.
-->

Discussion: <!-- Link to the GitHub Discussion -->

## Description

♻️ Do not accept underscore headers when using `convert_underscores=True` (the default)

This means it would only accept header names separated with hyphens by default, and to accept header names separated with underscores,
...[609 more chars]

## fastapi_15030 (fastapi/fastapi) gold: 9 file(s), 560 changed lines: docs_src/server_sent_events/tutorial001_py310.py, docs_src/server_sent_events/tutorial002_py310.py, docs_src/server_sent_events/tutorial003_py310.py, docs_src/server_sent_events/tutorial004_py310.py, docs_src/server_sent_events/tutorial005_py310.py, fastapi/openapi/utils.py, fastapi/responses.py, fastapi/routing.py, fastapi/sse.py; tests: tests/test_sse.py, tests/test_tutorial/test_server_sent_events/test_tutorial001.py, tests/test_tutorial/test_server_sent_events/test_tutorial002.py, tests/test_tutorial/test_server_sent_events/test_tutorial003.py, tests/test_tutorial/test_server_sent_events/test_tutorial004.py, tests/test_tutorial/test_server_sent_events/test_tutorial005.py
✨ Add support for Server Sent Events

✨ Add support for Server Sent Events

This will be available in FastAPI 0.135.0, released in the next hours.

## fastapi_14962 (fastapi/fastapi) gold: 3 file(s), 55 changed lines: docs_src/custom_response/tutorial010_py310.py, fastapi/_compat/v2.py, fastapi/routing.py; tests: tests/test_dump_json_fast_path.py, tests/test_tutorial/test_custom_response/test_tutorial001.py, tests/test_tutorial/test_custom_response/test_tutorial010.py
✨ Serialize JSON response with Pydantic (in Rust), when there's a Pydantic return type or response model

✨ Serialize JSON response with Pydantic (in Rust), when there's a Pydantic return type or response model

## fastapi_15280 (fastapi/fastapi) gold: 2 file(s), 72 changed lines: docs_src/vibe/tutorial001_py310.py, fastapi/applications.py; tests: tests/test_vibe.py
✨ Add support for `@app.vibe()`

✨ Add support for `@app.vibe()`

## fastapi_14964 (fastapi/fastapi) gold: 1 file(s), 52 changed lines: fastapi/responses.py; tests: tests/test_deprecated_responses.py, tests/test_orjson_response_class.py, tests/test_tutorial/test_custom_response/test_tutorial001.py, tests/test_tutorial/test_custom_response/test_tutorial001b.py
🗑️ Deprecate `ORJSONResponse` and `UJSONResponse`

🗑️ Deprecate `ORJSONResponse` and `UJSONResponse`

Now that there's better performance by default, with response models: https://github.com/fastapi/fastapi/pull/14962

This removes `ujson` and `orjson` from the `"fastapi[all]"` extras.

To use these responses, `ujson` or `orjson` need to be explicitly installed.

## fastapi_15023 (fastapi/fastapi) gold: 1 file(s), 26 changed lines: docs_src/stream_data/tutorial002_py310.py; tests: tests/test_tutorial/test_stream_data/test_tutorial002.py
📝 Update docs for responses and new stream with `yield`

📝 Update docs for responses and new stream with `yield`

## fastapi_14986 (fastapi/fastapi) gold: 2 file(s), 34 changed lines: fastapi/applications.py, fastapi/openapi/docs.py; tests: tests/test_openapi_cache_root_path.py, tests/test_swagger_ui_escape.py
♻️ Refactor logic to handle OpenAPI and Swagger UI escaping data

♻️ Refactor logic to handle OpenAPI and Swagger UI escaping data

OpenAPI, do not store `root_path` in servers: the only way this could be a problem is if there was a misconfigured proxy that somehow allowed an attacker client to set `x-forwarded-*` headers and passed them along. For a proxy (or server) to do this, it normally has to be intentionally/explicitly misconfigured. But again, doesn't hurt to have it there.

Escape Swagger UI configs: I wouldn't consider this really important, the Swagger UI logic takes only data from 
...[392 more chars]

## fastapi_14978 (fastapi/fastapi) gold: 3 file(s), 82 changed lines: docs_src/strict_content_type/tutorial001_py310.py, fastapi/applications.py, fastapi/routing.py; tests: tests/test_strict_content_type_app_level.py, tests/test_strict_content_type_nested.py, tests/test_strict_content_type_router_level.py, tests/test_tutorial/test_body/test_tutorial001.py, tests/test_tutorial/test_strict_content_type/test_tutorial001.py
🔒️ Add `strict_content_type` checking for JSON requests

🔒️ Add `strict_content_type` checking for JSON requests

## rich_4070 (Textualize/rich) gold: 9 file(s), 128 changed lines: rich/_emoji_replace.py, rich/console.py, rich/emoji.py, rich/logging.py, rich/protocol.py, rich/repr.py, rich/segment.py, rich/syntax.py, rich/theme.py; tests: tests/test_console.py, tests/test_syntax.py
perf: reduce Console and RichHandler import time by deferring unused imports

## E2E Results

| Import | master | This PR | Speedup |
|---|---|---|---|
| `from rich.console import Console` | 78.1ms | 52.2ms | **1.50x faster** |
| `from rich.logging import RichHandler` | 99.4ms | 56.9ms | **1.75x faster** |
| `import rich` | 18.2ms | 18.3ms | (already lean) |

## Summary

Defer module-level imports that are only needed in specific code paths, move annotation-only imports to `TYPE_CHECKING`, and remove dead code:

| File | Change | Savings |
|---|---|---|
| `logging.py` | Defer `Traceback` to `e
...[2496 more chars]

## rich_4077 (Textualize/rich) gold: 1 file(s), 3 changed lines: rich/file_proxy.py; tests: tests/test_file_proxy.py
proxy isatty

Fixes https://github.com/Textualize/rich/issues/4041

## rich_4079 (Textualize/rich) gold: 1 file(s), 7 changed lines: rich/markdown.py; tests: tests/test_markdown.py
Inline table code

Fixes https://github.com/Textualize/rich/issues/4038

## rich_4076 (Textualize/rich) gold: 1 file(s), 6 changed lines: rich/ansi.py; tests: tests/test_ansi.py
preserve newlines

Fixes https://github.com/Textualize/rich/issues/3577

## rich_4075 (Textualize/rich) gold: 1 file(s), 15 changed lines: rich/console.py; tests: tests/test_console.py
empty with end

- Fixes https://github.com/Textualize/rich/issues/3740
- Drop support for Python 3.8

## rich_4006 (Textualize/rich) gold: 1 file(s), 45 changed lines: rich/cells.py; tests: tests/test_cells.py
fix for infinite loop in split_graphemes

Fix for https://github.com/Textualize/rich/issues/3958

## fastapi_14258 (fastapi/fastapi) gold: 1 file(s), 4 changed lines: fastapi/routing.py; tests: tests/test_router_circular_import.py
✨ Show a clear error on attempt to include router into itself

Previously discussed in https://github.com/fastapi/fastapi/discussions/11424 with the PR at https://github.com/fastapi/fastapi/pull/11425

Since I made modifications in the fork of the repo, the PR was automatically closed. This PR serves to add the circular reference check again. Discussed again in https://github.com/fastapi/fastapi/discussions/14251

## fastapi_14851 (fastapi/fastapi) gold: 1 file(s), 157 changed lines: fastapi/routing.py; tests: tests/test_router_events.py
♻️ Re-implement `on_event` in FastAPI for compatibility with the next Starlette, while keeping backwards compatibility

♻️ Re-implement `on_event` in FastAPI for compatibility with the next Starlette, while keeping backwards compatibility

Once there's a better interface for `lifespan` or a comparable alternative, `on_event` and others will be deprecated and later removed. But while that new interface is available, I'm re-implementing it on the FastAPI side.

## fastapi_14873 (fastapi/fastapi) gold: 1 file(s), 21 changed lines: fastapi/routing.py; tests: tests/test_router_events.py
🐛 Fix `on_startup` and `on_shutdown` parameters of `APIRouter`

This fixes the issue reported in this comment: https://github.com/fastapi/fastapi/pull/14851#issuecomment-3863896385

```py
from fastapi import FastAPI


def s_start():
    print("SYNC START", flush=True)


app = FastAPI(on_startup=[s_start])


@app.get("/health")
async def health():
    return {"ok": True}
```

Startup event is not called because `APIRouter.on_startup` is set in `APIRouter.__init__` before calling `super().__init__` and  is overridden by `router.__init__` (https://github.com/Kludex/starlette/blob/e5b8a5d200504a26
...[56 more chars]

## fastapi_13920 (fastapi/fastapi) gold: 1 file(s), 4 changed lines: fastapi/dependencies/utils.py; tests: tests/test_dependency_pep695.py
✨ Add support for PEP695 `TypeAliasType`

This is a continuation of #11140 with the comment by @Viicos fixed by vendoring a copy of `is_typealiastype` from [`typing-inspection`](https://github.com/pydantic/typing-inspection).

## fastapi_14794 (fastapi/fastapi) gold: 1 file(s), 5 changed lines: fastapi/dependencies/utils.py; tests: tests/test_response_dependency.py
✨ Allow `Response` type hint as dependency annotation

## Summary
Fixes #10127

Using `Response` as a type hint with `Depends` would fail with an assertion error:
```python
def endpoint(response: Annotated[Response, Depends(modify_response)]):
    # AssertionError: Cannot specify \`Depends\` for type Response
```

## Changes
Modified the type annotation handling in `analyze_param()` to check for `Depends` before applying special type handling. When a `Depends` is specified, the dependency is called normally and its return value used.

## Usage
Now supports:
- `response: Annotated[Response, Dep
...[216 more chars]

## fastapi_14786 (fastapi/fastapi) gold: 1 file(s), 2 changed lines: fastapi/security/utils.py; tests: tests/test_security_http_base.py, tests/test_security_oauth2_authorization_code_bearer.py
🐛 Strip whitespaces from `Authorization` header credentials

## Description

Fixes whitespace handling in authorization header parameter extraction to comply with [RFC 6750](https://datatracker.ietf.org/doc/html/rfc6750#section-2.1).

## Problem

The current implementation preserves leading/trailing whitespace in the extracted credential parameter. While RFC 6750 allows one or more spaces (`1*SP`) between "Bearer" and the token, the token definition (`b64token`) doesn't include whitespace characters, meaning spaces should be treated as separators only.

This causes the extracted credentials to
...[1632 more chars]

## fastapi_14492 (fastapi/fastapi) gold: 1 file(s), 2 changed lines: docs_src/metadata/tutorial001_1_py39.py; tests: tests/test_tutorial/test_metadata/test_tutorial001_1.py
📝 Fix example of license identifier in documentation

Fix the Apache 2.0 license identifier to be `Apache-2.0`, not `MIT` (which is NOT an Apache 2.0 license).

## fastapi_14479 (fastapi/fastapi) gold: 1 file(s), 2 changed lines: fastapi/dependencies/utils.py; tests: tests/test_invalid_sequence_param.py
🚸 Improve error message for invalid query parameter type annotations

When I bumped FastAPI from 0.181.1 to 0.181.2, project started crashing on startup with AssertionError. The traceback didn’t provide any helpful context. After debugging, I found that issue was caused by incorrectly typed query parameters after the changes introduced in PR #12942 

This PR adds a clearer error message to assertion, indicating that query parameter type is incorrect, and updates tests to ensure that new message is present.

Below is an example of a wrongly typed query parameter that didn’t fail before 0.181.2,
...[1800 more chars]

## fastapi_14463 (fastapi/fastapi) gold: 1 file(s), 3 changed lines: fastapi/openapi/utils.py; tests: tests/test_additional_responses_union_duplicate_anyof.py
🐛 Fix OpenAPI duplication of `anyOf` refs for app-level responses with specified `content` and `model` as `Union`

## Summary

Fixes an issue where defining app-level `responses={...}` with a `Union` of Pydantic models **and** specifying `["content"][...]["examples"]` causes the generated OpenAPI schema to accumulate duplicate `$ref` entries in the `anyOf` array.



## Minimal reproducible example

<details open>
<summary> <code>app.py</code> </summary>

```python
from fastapi import FastAPI
from pydantic import BaseModel


class ModelA(BaseModel):
    a: str


class ModelB(BaseModel):
    b: 
...[9126 more chars]

## fastapi_12942 (fastapi/fastapi) gold: 1 file(s), 3 changed lines: fastapi/_compat.py; tests: tests/test_union_body_discriminator.py
🐛 Fix tagged discriminated union not recognized as body field

## Problem

When using a union type models with a tagged discriminator as an endpoint parameter, like the following:

```py
class Pie(BaseModel):
    time_to_cook: int
    num_ingredients: int


class ApplePie(Pie):
    fruit: Literal["apple"] = "apple"


class PumpkinPie(Pie):
    filling: Literal["pumpkin"] = "pumpkin"


def get_discriminator_value(v: Any) -> str:
    if isinstance(v, dict):
        return v.get("fruit", v.get("filling"))
    return getattr(v, "fruit", getattr(v, "filling", None))


Dessert = Annotated[
    Annot
...[747 more chars]

## fastapi_14099 (fastapi/fastapi) gold: 5 file(s), 477 changed lines: docs_src/dependencies/tutorial013_an_py310.py, docs_src/dependencies/tutorial014_an_py310.py, fastapi/applications.py, fastapi/middleware/asyncexitstack.py, fastapi/routing.py; tests: tests/test_dependency_after_yield_raise.py, tests/test_dependency_after_yield_streaming.py, tests/test_dependency_after_yield_websockets.py, tests/test_dependency_contextmanager.py, tests/test_route_scope.py, tests/test_tutorial/test_dependencies/test_tutorial008c.py
🐛 Fix support for `StreamingResponse`s with dependencies with `yield` or `UploadFile`s, close after the response is done

🐛 Fix support for `StreamingResponse`s with dependencies with `yield` or `UploadFile`s, close after the response is done

---

There's (again) an `AsyncExitStack` middleware, now used mainly to close objects after the response is done.

There's also a new wrapper duplicating some logic in Starlette but creating an `AsyncExitStack` for the exit code in dependencies with `yield`, for regular responses and WebSockets as well.

The exit code of dependencies with `yield` is now 
...[36 more chars]

## fastapi_14609 (fastapi/fastapi) gold: 20 file(s), 2047 changed lines: docs_src/request_form_models/tutorial002_pv1_an_py39.py, docs_src/request_form_models/tutorial002_pv1_py39.py, fastapi/_compat/__init__.py, fastapi/_compat/main.py, fastapi/_compat/may_v1.py, fastapi/_compat/model_field.py, fastapi/_compat/shared.py, fastapi/_compat/v1.py, fastapi/_compat/v2.py, fastapi/datastructures.py, fastapi/dependencies/utils.py, fastapi/encoders.py, fastapi/exceptions.py, fastapi/openapi/models.py, fastapi/openapi/utils.py, fastapi/param_functions.py, fastapi/params.py, fastapi/routing.py, fastapi/temp_pydantic_v1_params.py, fastapi/utils.py; tests: tests/test_compat.py, tests/test_compat_params_v1.py, tests/test_datetime_custom_encoder.py, tests/test_filter_pydantic_sub_model/app_pv1.py, tests/test_filter_pydantic_sub_model/test_filter_pydantic_sub_model_pv1.py, tests/test_get_model_definitions_formfeed_escape.py, tests/test_inherited_custom_class.py, tests/test_jsonable_encoder.py
➖ Drop support for `pydantic.v1`

➖ Drop support for `pydantic.v1`

When trying to use a `pydantic.v1` model with FastAPI, an error is raised.

## fastapi_14372 (fastapi/fastapi) gold: 2 file(s), 7 changed lines: fastapi/dependencies/utils.py, fastapi/params.py; tests: tests/test_depends_hashable.py
♻️ Make the result of `Depends()` and `Security()` hashable, as a workaround for other tools interacting with these internal parts

♻️ Make the result of `Depends()` and `Security()` hashable, as a workaround for other tools interacting with these internal parts

---

This comes from https://github.com/fastapi/fastapi/pull/14320

As a note, FastAPI doesn't really have official support for extending its internals in ways that are not documented in the docs.

So this is not a long-term commitment to keep some specific behavior, but a quick workaround to make things easier for current tools that 
...[544 more chars]

## fastapi_14349 (fastapi/fastapi) gold: 1 file(s), 12 changed lines: fastapi/_compat/v2.py; tests: tests/test_schema_ref_pydantic_v2.py
🐛 Fix handling of JSON Schema attributes named "$ref"

Fix handling of JSON Schema attributes named "$ref"

Related to: https://github.com/fastapi/fastapi/issues/14344

## fastapi_14512 (fastapi/fastapi) gold: 1 file(s), 52 changed lines: fastapi/_compat/v2.py; tests: tests/test_union_body_discriminator_annotated.py
🐛 Fix support for tagged union with discriminator inside of `Annotated` with `Body()`

🐛 Fix support for tagged union with discriminator inside of `Annotated` with `Body()`

Related to: https://github.com/fastapi/fastapi/discussions/14495, https://github.com/fastapi/fastapi/issues/14508

## fastapi_14487 (fastapi/fastapi) gold: 1 file(s), 7 changed lines: docs_src/handling_errors/tutorial004.py; tests: tests/test_tutorial/test_handling_errors/test_tutorial004.py
📝 Update docs about re-raising validation errors, do not include string as is to not leak information

📝 Update docs about re-raising validation errors, do not include string as is to not leak information

## fastapi_11355 (fastapi/fastapi) gold: 1 file(s), 11 changed lines: fastapi/dependencies/utils.py; tests: tests/test_stringified_annotations_simple.py
🐛 Fix evaluating stringified annotations in Python 3.10

I've bumped into the exact same issue as https://github.com/tiangolo/typer/discussions/598 so I solved it like https://github.com/tiangolo/typer/pull/721  :smile: 

I'll try to add tests soon but if you can do it faster, please feel free !!

## fastapi_14485 (fastapi/fastapi) gold: 1 file(s), 19 changed lines: fastapi/dependencies/utils.py; tests: tests/test_stringified_annotation_dependency.py
🐛 Fix support for `if TYPE_CHECKING`,  non-evaluated stringified annotations

🐛 Fix support for `if TYPE_CHECKING`,  non-evaluated stringified annotations

Related to https://github.com/fastapi/fastapi/discussions/14464, https://github.com/fastapi/fastapi/issues/14484, https://github.com/fastapi/fastapi/pull/11355

## fastapi_14482 (fastapi/fastapi) gold: 1 file(s), 32 changed lines: fastapi/_compat/v2.py; tests: tests/test_arbitrary_types.py
🐛 Fix handling arbitrary types when using `arbitrary_types_allowed=True`

🐛 Fix handling arbitrary types when using `arbitrary_types_allowed=True`

Related to https://github.com/fastapi/fastapi/discussions/14184, https://github.com/fastapi/fastapi/issues/14483



## fastapi_14306 (fastapi/fastapi) gold: 2 file(s), 137 changed lines: fastapi/exceptions.py, fastapi/routing.py; tests: tests/test_validation_error_context.py
🚸  Improve tracebacks by adding endpoint metadata

Today, when validation errors occur, error messages don't indicate what endpoint caused the problem, which can make debugging difficult. 

This PR adds endpoint context metadata to validation error messages to show the filepath (clickable in IDEs!) and line number, function name and HTTP method and route where applicable. The endpoint context is extracted once per endpoint and then cached to avoid any additional performance overhead on subsequent requests. Just to double check, I also ran a rough benchmark (100k requests):
```
Baseline (withou
...[2129 more chars]

## fastapi_14458 (fastapi/fastapi) gold: 1 file(s), 7 changed lines: fastapi/dependencies/models.py; tests: tests/test_dependency_class.py
🐛 Fix using class (not instance) dependency that has `__call__` method

As reported in https://github.com/fastapi/fastapi/discussions/14452, FastAPI 0.123.6 introduced a regression - if we use class (not instance) as dependency and this class has `__call__` method that is coroutine or generator, it is mistakenly treated as coroutine\generator (actually class is not coroutine\generator, instance of this class is).


So, to fix this we need to check if the callable is class before inspecting its `dunder_call` in `is_coroutine_callable`, `is_async_gen_callable`, and `is_gen_callable`.

## fastapi_14459 (fastapi/fastapi) gold: 3 file(s), 71 changed lines: fastapi/dependencies/models.py, fastapi/dependencies/utils.py, fastapi/openapi/utils.py; tests: tests/test_security_oauth2_authorization_code_bearer_scopes_openapi.py, tests/test_security_oauth2_authorization_code_bearer_scopes_openapi_simple.py
🐛 Fix OAuth2 scopes in OpenAPI in extra corner cases, parent dependency with scopes, sub-dependency security scheme without scopes

🐛 Fix OAuth2 scopes in OpenAPI in extra corner cases, parent dependency with scopes, sub-dependency security scheme without scopes, and the same security scheme used as another sub-dependency without scopes in a different path operation.

The dependency without scopes would be cached for OpenAPI and for the one with scopes, it would render without scopes.

This also refactors the internals of how dependencies are handled, and how scopes and security schemes are pa
...[209 more chars]

## fastapi_14448 (fastapi/fastapi) gold: 2 file(s), 106 changed lines: fastapi/dependencies/models.py, fastapi/dependencies/utils.py; tests: tests/test_dependency_wrapped.py
🐛 Fix support for functools wraps and partial combined, for async and regular functions and classes in path operations and dependencies

🐛 Fix support for functools wraps and partial combined, for async and regular functions and classes in path operations and dependencies

This also uses and extends the tests created by @YuriiMotov :raised_hands: 

Should fix / related to: https://github.com/fastapi/fastapi/issues/14444

The new tests simulate the multiple combinations of how decorators could affect dependencies and path operation functions (endpoints).

The logic to analyze if a function is a
...[612 more chars]

## fastapi_14455 (fastapi/fastapi) gold: 1 file(s), 13 changed lines: fastapi/openapi/utils.py; tests: tests/test_security_oauth2_authorization_code_bearer_scopes_openapi.py
🐛 Fix OpenAPI security scheme OAuth2 scopes declaration, deduplicate security schemes with different scopes

🐛 Fix OpenAPI security scheme OAuth2 scopes declaration, deduplicate security schemes with different scopes

This should fix https://github.com/fastapi/fastapi/issues/14454

## fastapi_9753 (fastapi/fastapi) gold: 1 file(s), 7 changed lines: fastapi/dependencies/models.py; tests: tests/test_dependency_partial.py
✨ Allow using dependables with `functools.partial()`

Discussion: https://github.com/tiangolo/fastapi/discussions/9744

Using `functools.partial()` with dependables feels like a very obvious idiom. `partial()` already works for dependables that are plain function, but it doesn't work consistently, partial of async and callable classes didn't work consistently.

This PR makes partial() work consistently in all combinations.

test_dependency_class.py have been renamed to test_dependency_types.py to reflect its bigger scope now.

There's quite a bit of repetition in the tests, part of me wants to
...[81 more chars]

## fastapi_14430 (fastapi/fastapi) gold: 1 file(s), 4 changed lines: fastapi/_compat/v2.py; tests: tests/test_compat.py
🐛 Fix optional sequence handling with new union syntax from Python 3.10

https://github.com/fastapi/fastapi/pull/14297 is only taking one union type into account. I can only recommend using https://github.com/pydantic/typing-inspection for these kind of checks (see https://typing-inspection.pydantic.dev/latest/usage/ for motivation).

## fastapi_5077 (fastapi/fastapi) gold: 1 file(s), 6 changed lines: fastapi/dependencies/utils.py; tests: tests/forward_reference_type.py, tests/test_wrapped_method_forward_reference.py
✨ Add support for wrapped functions (e.g. `@functools.wraps()`) used with forward references

Fixes #5065 where the `globalns` computation in `get_typed_signature` misses an edge case handled by `typing.get_type_hints`. This PR makes the logic similar to `get_type_hints`.

## Background

`functools.wraps` updates the annotations and type signature to match the wrapped function. However, it copies the `__annotations__` dict verbatim, including forward references (strings). `get_type_hints` handles this by [dereferencing the `__wrapped__` attribute](https://github.com/python/cpython/blob/576dd90
...[2039 more chars]

## fastapi_9555 (fastapi/fastapi) gold: 1 file(s), 22 changed lines: fastapi/dependencies/models.py; tests: tests/test_dependency_wrapped.py
✨ Handle wrapped dependencies

Use the unwrapped call in `solve_dependencies` to determine if a dependency is a coroutine or (async) generator.

Since non-async dependencies are run in a threadpool, it's generally preferably to use async dependencies. For dependencies that will not change during the lifetime of the server (like settings), it's preferable to use `functools.cache` to avoid duplicate work. There are two issues with this.

First is that `@cache` will cache the coroutine itself not the result, so the first call will succeed; however, subsequent calls will fail with `RuntimeError: c
...[2255 more chars]

## fastapi_14361 (fastapi/fastapi) gold: 1 file(s), 2 changed lines: fastapi/_compat/v2.py; tests: tests/test_schema_compat_pydantic_v2.py
🐛 Avoid accessing non-existing "$ref" key for Pydantic v2 compat remapping

cf Discussion #14265

## fastapi_13207 (fastapi/fastapi) gold: 1 file(s), 14 changed lines: fastapi/_compat/v2.py; tests: tests/test_computed_fields.py
🐛 Fix OpenAPI schema support for computed fields when using `separate_input_output_schemas=False`

This demonstrates how the output schema differs when the FastAPI app runs with different values for `separate_input_output_schemas` - namely, the computed fields are omitted.

Initially reported here: https://github.com/fastapi/fastapi/discussions/10342

## fastapi_14360 (fastapi/fastapi) gold: 1 file(s), 3 changed lines: fastapi/dependencies/utils.py; tests: tests/test_request_param_model_by_alias.py
🐛 Fix Query\Header\Cookie parameter model alias

When we declare Query\Header\Cookie parameter as model and specify `alias` for the field, there are currently 2 issues:
1. Value is extracted by alias, but it's put into `values` with the key `field.name` instead of `key.alias`. Then, when we attempt to validate the model passing `values`, it fails to find the value for that field (because it searches by alias) and we have validation error.
2. If we pass the value by name, we will correctly get validation error, but the `input` field will not contain the extra parameter (passed by name). That's 
...[149 more chars]

## fastapi_5624 (fastapi/fastapi) gold: 1 file(s), 4 changed lines: fastapi/dependencies/utils.py; tests: tests/test_security_scopes_dont_propagate.py
🐛 Fix hierarchical security scope propagation

see defect #5623

## fastapi_14419 (fastapi/fastapi) gold: 2 file(s), 62 changed lines: fastapi/dependencies/models.py, fastapi/dependencies/utils.py; tests: tests/test_security_scopes.py, tests/test_security_scopes_sub_dependency.py
🐛 Cache dependencies that don't use scopes and don't have sub-dependencies with scopes

🐛 Cache dependencies that don't use scopes and don't have sub-dependencies with scopes

Related to https://github.com/fastapi/fastapi/pull/9790, and https://github.com/fastapi/fastapi/discussions/6024

Uses the ideas in @YuriiMotov's diagrams here for the tests: https://github.com/fastapi/fastapi/discussions/6024#discussioncomment-8541913

---

Dependencies are cached based on the function. And also on the **scopes**, but (now) **only** when those scopes are used by this dependency or by a sub-dependency: t
...[81 more chars]

## fastapi_14301 (fastapi/fastapi) gold: 1 file(s), 5 changed lines: fastapi/dependencies/utils.py; tests: tests/test_dependency_yield_scope.py
🐛 Fix `Depends(func, scope='function')` for top level (parameterless) dependencies

**Summary**

This PR enables proper support for `scope="function"` on `APIRoute` dependencies by passing the dependency’s scope through to `get_dependant`. Previously, parameter-less dependencies created via `get_parameterless_sub_dependant` ignored `depends.scope`, which led to unexpected behavior for function-scoped dependencies.

**Motivation**

Users expect `scope="function"` to work as defined, even when registered at the route level. Because `depends.scope` wasn’t forwarded, function scope behaved like th
...[132 more chars]

## fastapi_14262 (fastapi/fastapi) gold: 10 file(s), 266 changed lines: docs_src/dependencies/tutorial008e.py, docs_src/dependencies/tutorial008e_an.py, docs_src/dependencies/tutorial008e_an_py39.py, fastapi/dependencies/models.py, fastapi/dependencies/utils.py, fastapi/exceptions.py, fastapi/param_functions.py, fastapi/params.py, fastapi/routing.py, fastapi/types.py; tests: tests/test_dependency_yield_scope.py, tests/test_dependency_yield_scope_websockets.py, tests/test_tutorial/test_dependencies/test_tutorial008e.py
✨ Add support for dependencies with scopes, support `scope="request"` for dependencies with `yield` that exit before the response is sent

✨ Add support for dependencies with scopes, support `scope="request"` for dependencies with `yield` that exit before the response is sent

This adds support for

* `Depends(func, scope="request")`, the default.
* `Depends(func, scope="function")`, early exit, after the function, but before sending the response back.

---

I'm considering an alternative name for the `scope` parameter of `mode`, to avoid any confusion with OAuth scopes, e.g. `Security(scopes=
...[1799 more chars]

## fastapi_14266 (fastapi/fastapi) gold: 1 file(s), 18 changed lines: fastapi/dependencies/utils.py; tests: tests/test_top_level_security_scheme_in_openapi.py
🐛 Fix security schemes in OpenAPI when added at the top level app

🐛 Fix security schemes in OpenAPI when added at the top level app

Ref: https://github.com/fastapi/fastapi/discussions/14263
Ref: https://github.com/fastapi/fastapi/issues/14271

## fastapi_14246 (fastapi/fastapi) gold: 1 file(s), 28 changed lines: fastapi/_compat/v2.py; tests: tests/test_no_schema_split.py, tests/test_pydantic_v1_v2_multifile/test_multifile.py
🐛 Fix separation of schemas with nested models introduced in 0.119.0

🐛 Fix separation of schemas with nested models introduced in 0.119.0

This should fix / related to https://github.com/fastapi/fastapi/issues/14247 (https://github.com/fastapi/fastapi/discussions/14177)

## fastapi_9425 (fastapi/fastapi) gold: 1 file(s), 2 changed lines: fastapi/dependencies/utils.py; tests: tests/test_return_none_stringified_annotations.py
✨  Allow `None` as return type for bodiless responses

Fixes #9424 

If the return annotation of the route function is exactly `"None"`, return `None` for the typed annotation. This allows the status code `204 No Content` to be used.

## fastapi_13713 (fastapi/fastapi) gold: 2 file(s), 31 changed lines: fastapi/applications.py, fastapi/openapi/utils.py; tests: tests/main.py, tests/test_application.py
✨ Add OpenAPI `external_docs` parameter to `FastAPI`

This PR adds an external_docs parameter to the __init__ method of the FastAPI class. Currently, the external_docs parameter is not being passed when generating the OpenAPI documentation, which I believe is an oversight. By including this parameter in the FastAPI class constructor, the external_docs field will be correctly included in the root of the generated OpenAPI JSON or YAML file.

Reasoning:

The external_docs field provides valuable external documentation links and should be part of the OpenAPI metadata. Including it at the root leve
...[152 more chars]

## fastapi_14077 (fastapi/fastapi) gold: 3 file(s), 11 changed lines: docs_src/handling_errors/tutorial005.py, fastapi/exception_handlers.py, fastapi/openapi/utils.py; tests: tests/test_enforce_once_required_parameter.py
⬆️ Upgrade Starlette supported version range to >=0.40.0,<0.49.0

Starlette 0.48.0 introduced `HTTP_422_UNPROCESSABLE_CONTENT` as a new name for `HTTP_422_UNPROCESSABLE_ENTITY` and deprecated the old name due to RFC 9110; see https://github.com/Kludex/starlette/pull/2939.

I updated all instances ~to the new name and increased the lower bound on the Starlette version~ to use the literal integer or string value `422`/`"422"` instead of either `HTTP_422_UNPROCESSABLE_CONTENT` or `HTTP_422_UNPROCESSABLE_ENTITY`, which makes FastAPI compatible with Starlette >=0.48 without deprecation warnings whi
...[73 more chars]

## rich_3718 (Textualize/rich) gold: 1 file(s), 7 changed lines: rich/panel.py; tests: tests/test_panel.py
fix(panel): fix title missing panel background

 Fix `Panel` title missing the panel background style.
    
This really just reverts the change in 7a38204. There's a history of issues related to the panel title styles, so I was careful to run all the examples in those issues to ensure there wasn't any regression.
    
Fixes #3569

## Type of changes

- [x] Bug fix
- [ ] New feature
- [ ] Documentation / docstrings
- [ ] Tests
- [ ] Other

## Checklist

- [x] I've run the latest [black](https://github.com/psf/black) with default args on new code.
- [x] I've updated CHANGELOG.md and CONTRIBUTORS
...[129 more chars]

## rich_3934 (Textualize/rich) gold: 2 file(s), 20 changed lines: rich/live.py, rich/live_render.py; tests: tests/test_live.py
empty live

Fixes https://github.com/Textualize/rich/issues/3796

## rich_3953 (Textualize/rich) gold: 1 file(s), 26 changed lines: rich/cells.py; tests: tests/test_cells.py
Fix ZWJ and edge cases

Fixes https://github.com/Textualize/rich/issues/3947
Fixes https://github.com/Textualize/rich/issues/3950

## rich_3944 (Textualize/rich) gold: 1 file(s), 2 changed lines: rich/cells.py; tests: tests/test_cells.py
fix fonts

Fixes https://github.com/Textualize/rich/issues/3943

## rich_3942 (Textualize/rich) gold: 2 file(s), 70 changed lines: rich/default_styles.py, rich/markdown.py; tests: tests/_card_render.py, tests/test_markdown.py, tests/test_markdown_no_hyperlinks.py
Update to markdown styles

Updates to Markdown styling

## rich_3938 (Textualize/rich) gold: 2 file(s), 45 changed lines: rich/console.py, rich/segment.py; tests: tests/test_segment.py, tests/test_text.py
fix background style with soft wrap

Fixes https://github.com/Textualize/rich/issues/3838

## rich_3882 (Textualize/rich) gold: 1 file(s), 2 changed lines: rich/prompt.py; tests: tests/test_prompt.py
Fix raw markup printed on prompt errors sometimes

## Type of changes

- [x] Bug fix
- [ ] New feature
- [ ] Documentation / docstrings
- [ ] Tests
- [ ] Other

## Checklist

- [x] I've run the latest [black](https://github.com/psf/black) with default args on new code.
- [ ] I've updated CHANGELOG.md and CONTRIBUTORS.md where appropriate (see note about typos above).
- [x] I've added tests for new code.
- [x] I accept that @willmcgugan may be pedantic in the code review.

## Description

PromptBase.on_validate_error() should explicitly ask for markup in case the console passed to it has markup
...[33 more chars]

## rich_3894 (Textualize/rich) gold: 1 file(s), 4 changed lines: rich/_inspect.py; tests: tests/test_inspect.py
Handle unusual __qualname__ in inspect

<!--
Please note that Rich isn't accepting any new features at this point.

If a feature can be implemented without modifying the core library, then
they should be released as a third-party module. I can accept updates to the
core library that make it easier to extend (think hooks).

Bugfixes are always welcome of course.

Sometimes it is not clear what is a feature and what is a bug fix.
If there is any doubt, please open a discussion first.

-->

<!--
*Are you contributing typo fixes?*

If your PR solely consists of typos, at least one must be in the d
...[1314 more chars]

## rich_3935 (Textualize/rich) gold: 1 file(s), 16 changed lines: rich/table.py; tests: tests/test_columns.py, tests/test_table.py
fix for padding width

Fixes https://github.com/Textualize/rich/issues/3871

## rich_3905 (Textualize/rich) gold: 1 file(s), 7 changed lines: rich/progress.py; tests: tests/test_progress.py
Update progress.py

Don't output extraneous blank line when progress display is disabled


<!--
Please note that Rich isn't accepting any new features at this point.

If a feature can be implemented without modifying the core library, then
they should be released as a third-party module. I can accept updates to the
core library that make it easier to extend (think hooks).

Bugfixes are always welcome of course.

Sometimes it is not clear what is a feature and what is a bug fix.
If there is any doubt, please open a discussion first.

-->

<!--
*Are you contributing typo fixes?*

If your PR sole
...[1021 more chars]

## rich_3930 (Textualize/rich) gold: 26 file(s), 12714 changed lines: rich/__main__.py, rich/_cell_widths.py, rich/_unicode_data/__init__.py, rich/_unicode_data/_versions.py, rich/_unicode_data/unicode10-0-0.py, rich/_unicode_data/unicode11-0-0.py, rich/_unicode_data/unicode12-0-0.py, rich/_unicode_data/unicode12-1-0.py, rich/_unicode_data/unicode13-0-0.py, rich/_unicode_data/unicode14-0-0.py, rich/_unicode_data/unicode15-0-0.py, rich/_unicode_data/unicode15-1-0.py, rich/_unicode_data/unicode16-0-0.py, rich/_unicode_data/unicode17-0-0.py, rich/_unicode_data/unicode4-1-0.py, rich/_unicode_data/unicode5-0-0.py, rich/_unicode_data/unicode5-1-0.py, rich/_unicode_data/unicode5-2-0.py, rich/_unicode_data/unicode6-0-0.py, rich/_unicode_data/unicode6-1-0.py, rich/_unicode_data/unicode6-2-0.py, rich/_unicode_data/unicode6-3-0.py, rich/_unicode_data/unicode7-0-0.py, rich/_unicode_data/unicode8-0-0.py, rich/_unicode_data/unicode9-0-0.py, rich/cells.py; tests: tests/test_cells.py, tests/test_text.py, tests/test_unicode_data.py
Handle graphemes

Rich has never handled multi-codepoint emoji, because terminals were highly inconsistent about how they were rendered. And there was no way to know in advance how wide a given collection of codepoints was going to be. 

This is technically still the case. However modern terminals are more consistent now, and they tend to follow unicode rules. Which means that Rich can also follow those rules and get it right more of the time.

This PR uses the data from https://github.com/jquast/wcwidth to calculate widths in the same way. I didn't use wcwidth directly, as Rich has more compl
...[612 more chars]

## fastapi_14791 (fastapi/fastapi) gold: 1 file(s), 2 changed lines: fastapi/openapi/utils.py; tests: tests/test_additional_properties.py, tests/test_additional_properties_bool.py, tests/test_additional_responses_custom_model_in_callback.py, tests/test_additional_responses_default_validationerror.py, tests/test_annotated.py, tests/test_application.py, tests/test_dependency_duplicates.py, tests/test_enforce_once_required_parameter.py, tests/test_extra_routes.py, tests/test_filter_pydantic_sub_model_pv2.py, tests/test_forms_single_param.py, tests/test_generate_unique_id_function.py, tests/test_get_model_definitions_formfeed_escape.py, tests/test_get_request_body.py, tests/test_include_router_defaults_overrides.py, tests/test_infer_param_optionality.py, tests/test_modules_same_name_body/test_main.py, tests/test_multi_body_errors.py, tests/test_multi_query_errors.py, tests/test_no_schema_split.py, tests/test_openapi_examples.py, tests/test_openapi_query_parameter_extension.py, tests/test_openapi_separate_input_output_schemas.py, tests/test_param_in_path_and_dependency.py, tests/test_param_include_in_schema.py, tests/test_put_no_body.py, tests/test_regex_deprecated_body.py, tests/test_regex_deprecated_params.py, tests/test_repeated_dependency_schema.py
🐛 Update `ValidationError` schema to include `input` and `ctx`

## Summary

Related to #10787

The OpenAPI `ValidationError` schema was outdated and didn't reflect the actual error format returned by Pydantic V2.

## Changes
Updated `validation_error_definition` in `fastapi/openapi/utils.py` to include:
- `input`: The actual input value that caused the validation error  
- `ctx`: Optional context dictionary with additional error information

## Testing
All existing tests pass with the updated schema. The test snapshots have been updated to reflect the new schema structure.

## fastapi_15800 (fastapi/fastapi) gold: 8 file(s), 654 changed lines: docs_src/frontend/tutorial001_py310.py, docs_src/frontend/tutorial002_py310.py, docs_src/frontend/tutorial003_py310.py, docs_src/frontend/tutorial004_py310.py, docs_src/frontend/tutorial005_py310.py, docs_src/frontend/tutorial006_py310.py, fastapi/applications.py, fastapi/routing.py; tests: tests/test_frontend.py
✨ Add support for `app.frontend("/", directory="dist")` and `router.frontend("/", directory="dist")`

## Pull Request

✨ Add support for `app.frontend("/", directory="dist")` and `router.frontend("/", directory="dist")`

<!--
Please start with a GitHub Discussion.

Once a team member asks you to open a PR, create it and link the discussion here.

Obvious typo fixes can be made in a PR without starting a discussion.
-->

Discussion: <!-- Link to the GitHub Discussion -->

## Description

<!-- Write the description of your PR here -->

## AI Disclaimer

Codex with GPT 5.5, long design iterations
...[598 more chars]

## fastapi_14371 (fastapi/fastapi) gold: 4 file(s), 62 changed lines: fastapi/_compat/v2.py, fastapi/dependencies/utils.py, fastapi/openapi/utils.py, fastapi/params.py; tests: tests/test_request_params/test_body/test_list.py, tests/test_request_params/test_body/test_optional_list.py, tests/test_request_params/test_body/test_optional_str.py, tests/test_request_params/test_body/test_required_str.py, tests/test_request_params/test_cookie/test_optional_str.py, tests/test_request_params/test_cookie/test_required_str.py, tests/test_request_params/test_file/test_list.py, tests/test_request_params/test_file/test_optional.py, tests/test_request_params/test_file/test_optional_list.py, tests/test_request_params/test_file/test_required.py, tests/test_request_params/test_form/test_list.py, tests/test_request_params/test_form/test_optional_list.py, tests/test_request_params/test_form/test_optional_str.py, tests/test_request_params/test_form/test_required_str.py, tests/test_request_params/test_header/test_list.py, tests/test_request_params/test_header/test_optional_list.py, tests/test_request_params/test_header/test_optional_str.py, tests/test_request_params/test_header/test_required_str.py, tests/test_request_params/test_path/test_required_str.py, tests/test_request_params/test_query/test_list.py, tests/test_request_params/test_query/test_optional_list.py, tests/test_request_params/test_query/test_optional_str.py, tests/test_request_params/test_query/test_required_str.py
🐛 Fix parameter aliases

This PR fixes the rest of failing tests in #14358.

---

**Note:** for `validation_aliases` only string value is currently supported. We can later think of supporting `AliasPath` and `AliasChoices`


**Will also fix issues:**

<details>

* https://github.com/fastapi/fastapi/discussions/8535
* https://github.com/fastapi/fastapi/discussions/10014

</details>

## fastapi_14605 (fastapi/fastapi) gold: 7 file(s), 44 changed lines: fastapi/dependencies/utils.py, fastapi/exceptions.py, fastapi/openapi/utils.py, fastapi/params.py, fastapi/routing.py, fastapi/temp_pydantic_v1_params.py, fastapi/utils.py; tests: tests/benchmarks/test_general_performance.py, tests/test_compat_params_v1.py, tests/test_pydantic_v1_deprecation_warnings.py, tests/test_regex_deprecated_body.py, tests/test_regex_deprecated_params.py, tests/test_schema_extra_examples.py, tests/test_tutorial/test_pydantic_v1_in_v2/test_tutorial002.py, tests/test_tutorial/test_pydantic_v1_in_v2/test_tutorial003.py, tests/test_tutorial/test_pydantic_v1_in_v2/test_tutorial004.py, tests/test_tutorial/test_query_params_str_validations/test_tutorial004.py, tests/test_tutorial/test_request_form_models/test_tutorial002_pv1.py, tests/test_tutorial/test_schema_extra_example/test_tutorial001_pv1.py
🔊 Add a custom `FastAPIDeprecationWarning`

🔊 Add a custom `FastAPIDeprecationWarning` and use it for deprecated logic, like Pydantic v1 models.

Motivated by: https://sethmlarson.dev/deprecations-via-warnings-dont-work-for-python-libraries

## fastapi_14583 (fastapi/fastapi) gold: 2 file(s), 24 changed lines: fastapi/dependencies/utils.py, fastapi/routing.py; tests: tests/benchmarks/test_general_performance.py, tests/test_compat_params_v1.py, tests/test_datetime_custom_encoder.py, tests/test_filter_pydantic_sub_model/app_pv1.py, tests/test_get_model_definitions_formfeed_escape.py, tests/test_pydantic_v1_deprecation_warnings.py, tests/test_pydantic_v1_v2_01.py, tests/test_pydantic_v1_v2_list.py, tests/test_pydantic_v1_v2_mixed.py, tests/test_pydantic_v1_v2_multifile/main.py, tests/test_pydantic_v1_v2_noneable.py, tests/test_read_with_orm_mode.py, tests/test_response_model_as_return_annotation.py, tests/test_tutorial/test_pydantic_v1_in_v2/test_tutorial002.py, tests/test_tutorial/test_pydantic_v1_in_v2/test_tutorial003.py, tests/test_tutorial/test_pydantic_v1_in_v2/test_tutorial004.py, tests/test_tutorial/test_request_form_models/test_tutorial002_pv1.py, tests/test_tutorial/test_schema_extra_example/test_tutorial001_pv1.py
🔊 Add deprecation warnings when using `pydantic.v1`

🔊 Add deprecation warnings when using `pydantic.v1`

## fastapi_14297 (fastapi/fastapi) gold: 1 file(s), 7 changed lines: fastapi/_compat/v2.py; tests: tests/test_compat.py, tests/test_optional_file_list.py
🐛 Fix optional sequence handling in `serialize sequence value` with Pydantic V2

Currently the following test code leads to `TypeError: issubclass() arg 1 must be a class` (with Python 3.14 error is different, but the root cause is the same):

```py
from typing import List, Optional

from fastapi import FastAPI, File
from fastapi.testclient import TestClient

app = FastAPI()


@app.post("/files")
async def upload_files(files: Optional[List[bytes]] = File(None)):
    if files is None:
        return {"files_count": 0}
    return {"files_count": len(files), "sizes": [len(f) for f in files]}


de
...[587 more chars]

## fastapi_14303 (fastapi/fastapi) gold: 1 file(s), 8 changed lines: fastapi/dependencies/utils.py; tests: tests/test_forms_single_model.py
🐛 Fix parsing extra `Form` parameter list

## Description

Currently if we declare Form parameters as model with `extra="allow"` and then pass extra parameter with multiple values, only last value of such parameter will be passed to model validator:

```py
from fastapi import FastAPI, Form
from fastapi.testclient import TestClient
from pydantic import BaseModel

app = FastAPI()


class Model(BaseModel):
    param: str

    model_config = {"extra": "allow"}

@app.post("/")
async def model_with_extra(data: Model = Form()):
    return data


def test_pass_extra_list():
    client = TestClient(app
...[1957 more chars]

## fastapi_13537 (fastapi/fastapi) gold: 1 file(s), 3 changed lines: fastapi/dependencies/utils.py; tests: tests/test_form_default.py
🐛 Fix support for form values with empty strings interpreted as missing (`None` if that's the default), for compatibility with HTML forms

There was a bug in the validation of form data for optional params with None default value. This is a regression that was introduced by https://github.com/fastapi/fastapi/pull/12134.

## Explaination:
The `_get_multidict_value` uses `None` to represent both the absence a value in the dict, and a default value of `None` for that field. When the values are collected in `_extract_form_body`, they are only added to the values if they are not `None`. Later (#121
...[1042 more chars]

## fastapi_14953 (fastapi/fastapi) gold: 4 file(s), 104 changed lines: docs_src/json_base64_bytes/tutorial001_py310.py, fastapi/_compat/v2.py, fastapi/datastructures.py, scripts/playwright/json_base64_bytes/image01.py; tests: tests/test_request_params/test_file/test_list.py, tests/test_request_params/test_file/test_optional.py, tests/test_request_params/test_file/test_optional_list.py, tests/test_request_params/test_file/test_required.py, tests/test_tutorial/test_json_base64_bytes/test_tutorial001.py, tests/test_tutorial/test_request_files/test_tutorial001.py, tests/test_tutorial/test_request_files/test_tutorial001_02.py, tests/test_tutorial/test_request_files/test_tutorial001_03.py, tests/test_tutorial/test_request_files/test_tutorial002.py, tests/test_tutorial/test_request_files/test_tutorial003.py, tests/test_tutorial/test_request_forms_and_files/test_tutorial001.py
♻️ Fix JSON Schema for bytes, use `"contentMediaType": "application/octet-stream"` instead of `"format": "binary"`

♻️ Fix JSON Schema for bytes, use `"contentMediaType": "application/octet-stream"` instead of `"format": "binary"`

## Background

`format: binary` was defined in OpenAPI 3.0.x, in OpenAPI 3.1.x the schema was aligned with the latest JSON Schema, recommending instead `contentMediaType: application/octet-stream`.

I suspect the JSON Schema for `bytes` using `"format": "binary"` comes from my first implementation in Pydantic 1.x.

It was defined and suggested in OpenAPI 3.0.x (not 
...[1151 more chars]

## fastapi_13786 (fastapi/fastapi) gold: 11 file(s), 259 changed lines: docs_src/authentication_error_status_code/tutorial001_an.py, docs_src/authentication_error_status_code/tutorial001_an_py39.py, docs_src/security/tutorial003.py, docs_src/security/tutorial003_an.py, docs_src/security/tutorial003_an_py310.py, docs_src/security/tutorial003_an_py39.py, docs_src/security/tutorial003_py310.py, fastapi/security/api_key.py, fastapi/security/http.py, fastapi/security/oauth2.py, fastapi/security/open_id_connect_url.py; tests: tests/test_security_api_key_cookie.py, tests/test_security_api_key_cookie_description.py, tests/test_security_api_key_header.py, tests/test_security_api_key_header_description.py, tests/test_security_api_key_query.py, tests/test_security_api_key_query_description.py, tests/test_security_http_base.py, tests/test_security_http_base_description.py, tests/test_security_http_basic_optional.py, tests/test_security_http_basic_realm.py, tests/test_security_http_basic_realm_description.py, tests/test_security_http_bearer.py, tests/test_security_http_bearer_description.py, tests/test_security_http_digest.py, tests/test_security_http_digest_description.py, tests/test_security_oauth2.py, tests/test_security_openid_connect.py
🐛 Use `401` status code in security classes when credentials are missing

## Warning

This description is partially outdated after changes described in [this comment](https://github.com/fastapi/fastapi/pull/13786#pullrequestreview-3501695067).

## Description

This PR is an attempt to finally solve the issue with security tools returning error responses with status code `403` instead of `401` when credentials are not provided.

## Breaking changes and workaround

These changes can break projects that rely on old behavior.
In order to mitigate this, the `not_authenticated_status_code` is introd
...[4651 more chars]

## fastapi_14616 (fastapi/fastapi) gold: 1 file(s), 12 changed lines: fastapi/dependencies/utils.py; tests: tests/test_json_type.py
🐛 Fix using `Json[list[str]]` type (issue #10997)

Fixes Form() not working with Pydantic `Json[T]` types when T is a sequence type (list, set, tuple).

Closes #10997

When using `Form()` with `Json[list[str]]`, FastAPI incorrectly treated it as a sequence field and used `getlist()`, wrapping the JSON string in a list. This caused Pydantic to fail parsing with:
```
TypeError: JSON input should be string, bytes or bytearray
```

**Before fix:**
```python
# Client sends
data = {"items": '["abc", "def"]'}  # JSON string

# FastAPI incorrectly does
value = values.getlist("items")  # ['["abc", "def
...[538 more chars]

## fastapi_11194 (fastapi/fastapi) gold: 1 file(s), 7 changed lines: fastapi/dependencies/utils.py; tests: tests/test_file_and_form_order_issue_9116.py
🐛 Fix validation error when `File` is declared after `Form` parameter

This PR updates the fix (6185a92) from @jsg921019 so that tests pass with the latest FastAPI code (0.110.0).
It also adds a test to cover the case with multiple files, and another to ensure the order of parameters in an endpoint definition does not affect the way FastAPI handles the request.

**Problem**
When declaring an endpoint which accepts a `File` but also a `Form` parameter, the `File` parameter must be declared first or the endpoint would respond with error 422.

**Context**
- See discussion #9116
- And probably als
...[8 more chars]

## requests_6644 (psf/requests) gold: 1 file(s), 3 changed lines: src/requests/adapters.py; tests: tests/test_adapters.py
Trim excess leading path separators

A URL with excess leading / (path-separator)s would cause urllib3 to attempt to reparse the request-uri as a full URI with a host and port. This bypasses that logic in ConnectionPool.urlopen by replacing these leading /s with just a single /.

Closes #6643

## rich_3180 (Textualize/rich) gold: 2 file(s), 127 changed lines: rich/_wrap.py, rich/cells.py; tests: tests/test_cells.py, tests/test_text.py
Fix double-width characters disappearing when wrapping

## Type of changes

- [x] Bug fix
- [ ] New feature
- [ ] Documentation / docstrings
- [ ] Tests
- [ ] Other

## Checklist

- [x] I've run the latest [black](https://github.com/psf/black) with default args on new code.
- [x] I've updated CHANGELOG.md and CONTRIBUTORS.md where appropriate.
- [x] I've added tests for new code.
- [x] I accept that @willmcgugan may be pedantic in the code review.

## Description

Update wrapping logic to fix issues with CJK charcters disappearing when the "fold" location sat *within* a double-width character.
...[563 more chars]

## rich_3777 (Textualize/rich) gold: 2 file(s), 13 changed lines: rich/console.py, rich/diagnose.py; tests: tests/test_console.py
TTY_INTERACTIVE env var

Adds a `TTY_INTERACTIVE` to disable things like progress bar / status spinners in CI

This adds to the new `TTY_COMPATIBLE` which enables styles even when the file is not technically a tty, as is the case in Github Actions. `TTY_INTERACTIVE=0` will remove superfluous animations and progress bars in the CI output.

## rich_3468 (Textualize/rich) gold: 1 file(s), 42 changed lines: rich/console.py; tests: tests/test_console.py
handle broken pipe

Handle BrokenPipe exceptions

Fixes https://github.com/Textualize/rich/issues/1591

Thanks to @lmmx for doing the legwork.

## rich_3782 (Textualize/rich) gold: 1 file(s), 29 changed lines: rich/syntax.py; tests: tests/test_syntax.py
Syntax padding

Fixes https://github.com/Textualize/rich/issues/3727

## rich_3772 (Textualize/rich) gold: 1 file(s), 45 changed lines: rich/traceback.py; tests: tests/test_traceback.py
Fix traceback recursion

Fixes https://github.com/Textualize/rich/issues/3708
Fixes https://github.com/Textualize/rich/issues/3682

## rich_3675 (Textualize/rich) gold: 2 file(s), 41 changed lines: rich/console.py, rich/diagnose.py; tests: tests/test_console.py
TTY_COMPATIBLE

Implement new `TTY_COMPATIBLE` environment variable

@ofek @webknjaz @hugovk 

Relevant discussion: https://github.com/Textualize/rich/issues/2924


## rich_3676 (Textualize/rich) gold: 2 file(s), 13 changed lines: rich/default_styles.py, rich/traceback.py; tests: tests/test_traceback.py
Notes

Add support for Exception.__notes__

## rich_3518 (Textualize/rich) gold: 1 file(s), 2 changed lines: rich/table.py; tests: tests/test_table.py
fix(table): highlight columns added by add_row

## Type of changes

- [x] Bug fix
- [ ] New feature
- [ ] Documentation / docstrings
- [ ] Tests
- [ ] Other

## Checklist

- [x] I've run the latest [black](https://github.com/psf/black) with default args on new code.
- [x] I've updated CHANGELOG.md and CONTRIBUTORS.md where appropriate.
- [x] I've added tests for new code.
- [x] I accept that @willmcgugan may be pedantic in the code review.

## Description

Fixes #3517. The default value for the `Column.highlight` is False, so where necessary columns are added by the `add_row` method this shoul
...[41 more chars]

## rich_3535 (Textualize/rich) gold: 2 file(s), 8 changed lines: rich/cells.py, rich/segment.py; tests: tests/test_cells.py
Regex error

Fixes an issue with the regex used to select the fast path for strings with single-width characters.

This may have had a detrimental effect on performance for some operations.


## rich_3521 (Textualize/rich) gold: 1 file(s), 35 changed lines: rich/segment.py; tests: tests/test_segment.py
Segment.split_cells fix

Fixes https://github.com/Textualize/textual/issues/5090

## rich_3506 (Textualize/rich) gold: 1 file(s), 17 changed lines: rich/segment.py; tests: tests/test_segment.py
fix split cells

Fixed `Segment._split_cells` issue discovered in https://github.com/Textualize/textual/issues/4996

## rich_3480 (Textualize/rich) gold: 1 file(s), 4 changed lines: rich/text.py; tests: tests/test_text.py
fix infinite loop in append

Fixes https://github.com/Textualize/rich/issues/3479

## rich_3472 (Textualize/rich) gold: 1 file(s), 4 changed lines: rich/pretty.py; tests: tests/test_pretty.py
fix for missing field in dataclass

Fixes https://github.com/Textualize/rich/issues/3417

## rich_3454 (Textualize/rich) gold: 1 file(s), 2 changed lines: rich/highlighter.py; tests: tests/test_highlighter.py
Fix [BUG] @ breaks highlighting in hyperlink #3327

## Type of changes

- [x] Bug fix
- [ ] New feature
- [ ] Documentation / docstrings
- [x] Tests
- [ ] Other

## Checklist

- [ ] I've run the latest [black](https://github.com/psf/black) with default args on new code.
- [ ] I've updated CHANGELOG.md and CONTRIBUTORS.md where appropriate.
- [x] I've added tests for new code.
- [x] I accept that @willmcgugan may be pedantic in the code review.

## Description
Fixed the bug [#3327](https://github.com/Textualize/rich/issues/3327).

Changes -
1. Added '@' in rich.highlighter:ReprHighlighter url r
...[242 more chars]

## rich_3471 (Textualize/rich) gold: 1 file(s), 1 changed lines: rich/text.py; tests: tests/test_text.py
Fixed exception in append_tokens

Fixes https://github.com/Textualize/rich/issues/3014

## rich_3470 (Textualize/rich) gold: 1 file(s), 2 changed lines: rich/console.py; tests: tests/test_console.py
Fix record and capture

Fix https://github.com/Textualize/rich/issues/2563

## rich_3052 (Textualize/rich) gold: 1 file(s), 34 changed lines: rich/prompt.py; tests: tests/test_prompt.py
Case insensitive prompt choices

## Type of changes

- [ ] Bug fix
- [x] New feature
- [ ] Documentation / docstrings
- [ ] Tests
- [ ] Other

## Checklist

- [x] I've run the latest [black](https://github.com/psf/black) with default args on new code.
- [x] I've updated CHANGELOG.md and CONTRIBUTORS.md where appropriate.
- [x] I've added tests for new code.
- [x] I accept that @willmcgugan may be pedantic in the code review.

## Description

Optional flag for `Prompt.ask` allowing case-insensitive responses. Just add `case_sensitive=False` to the command options:

```python
from rich.prompt im
...[236 more chars]

## rich_3278 (Textualize/rich) gold: 1 file(s), 1 changed lines: rich/ansi.py; tests: tests/test_ansi.py
Strip problematic private escape sequences

## Type of changes

- [x] Bug fix
- [ ] New feature
- [ ] Documentation / docstrings
- [ ] Tests
- [ ] Other

## Checklist

- [x] I've run the latest [black](https://github.com/psf/black) with default args on new code.
- [x] I've updated CHANGELOG.md and CONTRIBUTORS.md where appropriate.
- [x] I've added tests for new code.
- [x] I accept that @willmcgugan may be pedantic in the code review.

## Description

["Private escape sequences"](https://en.wikipedia.org/wiki/ANSI_escape_code#Fp_Escape_sequences) are escape sequences that are reserved for pri
...[595 more chars]

## rich_2943 (Textualize/rich) gold: 1 file(s), 2 changed lines: rich/style.py; tests: tests/test_style.py
Clear hashed cache when clearing meta.

This fixes #2942.

## Type of changes

- [x] Bug fix
- [ ] New feature
- [ ] Documentation / docstrings
- [ ] Tests
- [ ] Other

## Checklist

- [x] I've run the latest [black](https://github.com/psf/black) with default args on new code.
- [x] I've updated CHANGELOG.md and CONTRIBUTORS.md where appropriate.
- [x] I've added tests for new code.
- [x] I accept that @willmcgugan may be pedantic in the code review.



## rich_2725 (Textualize/rich) gold: 1 file(s), 10 changed lines: rich/table.py; tests: tests/test_table.py
fix table rendering order of box elements

## Type of changes

- [x] Bug fix
- [ ] New feature
- [ ] Documentation / docstrings
- [x] Tests
- [ ] Other

## Checklist

- [x] I've run the latest [black](https://github.com/psf/black) with default args on new code.
- [x] I've updated CHANGELOG.md and CONTRIBUTORS.md where appropriate.
- [x] I've added tests for new code.
- [x] I accept that @willmcgugan may be pedantic in the code review.

## Description

* Previously "mid" box elements appeared in last line of table (footer), while "footer" box elements appeared mid table body. Reversed that orde
...[1132 more chars]

## rich_3105 (Textualize/rich) gold: 1 file(s), 2 changed lines: rich/_export_format.py; tests: tests/test_console.py
Fix #3104

## Type of changes

- [X] Bug fix
- [ ] New feature
- [ ] Documentation / docstrings
- [ ] Tests
- [ ] Other

## Checklist

- [ ] I've run the latest [black](https://github.com/psf/black) with default args on new code.
- [ ] I've updated CHANGELOG.md and CONTRIBUTORS.md where appropriate.
- [ ] I've added tests for new code.
- [X] I accept that @willmcgugan may be pedantic in the code review.

## Description

See #3104


## rich_3043 (Textualize/rich) gold: 1 file(s), 2 changed lines: rich/_export_format.py; tests: tests/test_console.py
Fix the HTML export template

This PR fixes the HTML export template, addressing the issue raised in #3021

In doing so there's also an update to the pre-commit configuration (without it I wasn't able to commit the change -- the update is stolen from the same config file in Textual).

## rich_3067 (Textualize/rich) gold: 1 file(s), 2 changed lines: rich/highlighter.py; tests: tests/test_highlighter.py
Allow tilde in URL (fixes #3057)

This PR fixes #3057.

## Type of changes

- [X] Bug fix
- [ ] New feature
- [ ] Documentation / docstrings
- [x] Tests
- [ ] Other

## Checklist

- [x] I've run the latest [black](https://github.com/psf/black) with default args on new code.
- [X] I've updated CHANGELOG.md and CONTRIBUTORS.md where appropriate.
- [x] I've added tests for new code.
- [X] I accept that @willmcgugan may be pedantic in the code review.

## Description

Please describe your changes here. If this fixes a bug, please link to the issue, if possible.


## rich_3006 (Textualize/rich) gold: 1 file(s), 2 changed lines: rich/repr.py; tests: tests/test_repr.py
Fixed issue with custom classes (fixes #2875)

## Type of changes

- [x] Bug fix
- [ ] New feature
- [ ] Documentation / docstrings
- [ ] Tests
- [ ] Other

## Checklist

- [x] I've run the latest [black](https://github.com/psf/black) with default args on new code.
- [x] I've updated CHANGELOG.md and CONTRIBUTORS.md where appropriate.
- [x] I've added tests for new code.
- [x] I accept that @willmcgugan may be pedantic in the code review.

## Description

When `repr`ing classes that have overwritten `__eq__` as parameters, the check with `== Parameters.empty` can be `True`.
Fixes #2875 with th
...[58 more chars]

## rich_3063 (Textualize/rich) gold: 1 file(s), 4 changed lines: rich/markup.py; tests: tests/test_markup.py
Fix escaping trailing backslash

Fixes https://github.com/Textualize/rich/issues/2987

## rich_3061 (Textualize/rich) gold: 1 file(s), 91 changed lines: rich/text.py; tests: tests/test_panel.py, tests/test_text.py
Extend tabs with styles

The `Text.extend_tabs` method was previously not taking in to account the spans.  Basically it would replace tabs with unstyled spaces. This resolves that issue by extending the style(s) on the tab to cover the new space characters

<img width="623" alt="Screenshot 2023-07-28 at 22 00 06" src="https://github.com/Textualize/rich/assets/554369/28c170a0-b154-4a81-af17-a8a334bebd3a">


