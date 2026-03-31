from app.adapters.decision import DecisionModelAdapter, DecisionOutputParseError


def test_decision_parser_accepts_json_object() -> None:
    parsed = DecisionModelAdapter.parse(
        '{"decision":"SPEAK","intent":"continue","window":"short","reason":"unresolved plan","confidence":0.84,"urgency":"medium","suggested_topic":"next steps"}'
    )
    assert parsed.decision == "SPEAK"
    assert parsed.intent == "continue"
    assert parsed.window == "short"
    assert parsed.reason == "unresolved plan"
    assert parsed.confidence == 0.84
    assert parsed.suggested_topic == "next steps"


def test_decision_parser_extracts_embedded_json() -> None:
    parsed = DecisionModelAdapter.parse(
        "Here is the result:\n{\"decision\":\"WAIT\",\"reason\":\"question already asked\",\"confidence\":0.91}"
    )
    assert parsed.decision == "WAIT"
    assert parsed.reason == "question already asked"


def test_decision_parser_rejects_invalid_decision() -> None:
    try:
        DecisionModelAdapter.parse('{"decision":"MAYBE","reason":"bad","confidence":0.1}')
    except DecisionOutputParseError as exc:
        assert "invalid decision" in str(exc)
    else:
        raise AssertionError("parser should reject invalid decisions")
