"""Smoke tests for tasks/text/ — verify importability and basic execution."""

import pytest


# ---------------------------------------------------------------------------
# build_chat_prompt
# ---------------------------------------------------------------------------
class TestBuildChatPromptSmoke:
    """Smoke tests for tasks.text.build_chat_prompt."""

    def test_importable(self):
        from tasks.text.build_chat_prompt import build_chat_prompt

        assert hasattr(build_chat_prompt, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.text.build_chat_prompt import build_chat_prompt

        result = build_chat_prompt.__wrapped_function__(user_message="Hello")
        assert "Hello" in result
        assert isinstance(result, str)

    @pytest.mark.unit
    def test_with_system_message(self):
        from tasks.text.build_chat_prompt import build_chat_prompt

        result = build_chat_prompt.__wrapped_function__(
            user_message="Hello", system_message="You are helpful."
        )
        assert "Hello" in result
        assert "You are helpful." in result


# ---------------------------------------------------------------------------
# char_count
# ---------------------------------------------------------------------------
class TestCharCountSmoke:
    """Smoke tests for tasks.text.character_count."""

    def test_importable(self):
        from tasks.text.character_count import char_count

        assert hasattr(char_count, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.text.character_count import char_count

        result = char_count.__wrapped_function__("hello")
        assert result == 5

    @pytest.mark.unit
    def test_exclude_spaces(self):
        from tasks.text.character_count import char_count

        result = char_count.__wrapped_function__("hello world", exclude_spaces=True)
        assert result == 10


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------
class TestChunkTextSmoke:
    """Smoke tests for tasks.text.chunk_text."""

    def test_importable(self):
        from tasks.text.chunk_text import chunk_text

        assert hasattr(chunk_text, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.text.chunk_text import chunk_text

        chunks, count = chunk_text.__wrapped_function__("hello world", chunk_size=100)
        assert isinstance(chunks, list)
        assert len(chunks) == count
        assert count >= 1

    @pytest.mark.unit
    def test_empty_text(self):
        from tasks.text.chunk_text import chunk_text

        chunks, count = chunk_text.__wrapped_function__("")
        assert chunks == []
        assert count == 0


# ---------------------------------------------------------------------------
# concatenate
# ---------------------------------------------------------------------------
class TestConcatenateSmoke:
    """Smoke tests for tasks.text.concatenate."""

    def test_importable(self):
        from tasks.text.concatenate import concatenate

        assert hasattr(concatenate, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.text.concatenate import concatenate

        result = concatenate.__wrapped_function__(text_1="hello", text_2=" world")
        assert result == "hello world"

    @pytest.mark.unit
    def test_with_separator(self):
        from tasks.text.concatenate import concatenate

        result = concatenate.__wrapped_function__(text_1="a", text_2="b", separator="-")
        assert result == "a-b"


# ---------------------------------------------------------------------------
# extract_pattern
# ---------------------------------------------------------------------------
class TestExtractPatternSmoke:
    """Smoke tests for tasks.text.extract_pattern."""

    def test_importable(self):
        from tasks.text.extract_pattern import extract_pattern

        assert hasattr(extract_pattern, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.text.extract_pattern import extract_pattern

        result = extract_pattern.__wrapped_function__("hello world", r"\w+")
        assert result == "hello"

    @pytest.mark.unit
    def test_no_match_returns_default(self):
        from tasks.text.extract_pattern import extract_pattern

        result = extract_pattern.__wrapped_function__("hello", r"\d+", default="none")
        assert result == "none"


# ---------------------------------------------------------------------------
# find_all_patterns
# ---------------------------------------------------------------------------
class TestFindAllPatternsSmoke:
    """Smoke tests for tasks.text.find_all_patterns."""

    def test_importable(self):
        from tasks.text.find_all_patterns import find_all_patterns

        assert hasattr(find_all_patterns, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.text.find_all_patterns import find_all_patterns

        result = find_all_patterns.__wrapped_function__("hello world", r"\w+")
        assert result == ["hello", "world"]

    @pytest.mark.unit
    def test_no_match(self):
        from tasks.text.find_all_patterns import find_all_patterns

        result = find_all_patterns.__wrapped_function__("hello", r"\d+")
        assert result == []


# ---------------------------------------------------------------------------
# first_word
# ---------------------------------------------------------------------------
class TestFirstWordSmoke:
    """Smoke tests for tasks.text.first_word."""

    def test_importable(self):
        from tasks.text.first_word import first_word

        assert hasattr(first_word, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.text.first_word import first_word

        result = first_word.__wrapped_function__("hello world")
        assert result == "hello"

    @pytest.mark.unit
    def test_preserve_case(self):
        from tasks.text.first_word import first_word

        result = first_word.__wrapped_function__("Hello World", lowercase=False)
        assert result == "Hello"


# ---------------------------------------------------------------------------
# format_template
# ---------------------------------------------------------------------------
class TestFormatTemplateSmoke:
    """Smoke tests for tasks.text.format_template."""

    def test_importable(self):
        from tasks.text.format_template import format_template

        assert hasattr(format_template, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.text.format_template import format_template

        result = format_template.__wrapped_function__("{name}", values={"name": "world"})
        assert result == "world"

    @pytest.mark.unit
    def test_multiple_placeholders(self):
        from tasks.text.format_template import format_template

        result = format_template.__wrapped_function__(
            "Hello {name}!", values={"name": "Alice"}
        )
        assert result == "Hello Alice!"


# ---------------------------------------------------------------------------
# join_list
# ---------------------------------------------------------------------------
class TestJoinListSmoke:
    """Smoke tests for tasks.text.join_list."""

    def test_importable(self):
        from tasks.text.join_list import join_list

        assert hasattr(join_list, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.text.join_list import join_list

        result = join_list.__wrapped_function__(["a", "b", "c"], separator=", ")
        assert result == "a, b, c"

    @pytest.mark.unit
    def test_default_separator(self):
        from tasks.text.join_list import join_list

        result = join_list.__wrapped_function__(["a", "b"])
        assert result == "a\nb"


# ---------------------------------------------------------------------------
# last_word
# ---------------------------------------------------------------------------
class TestLastWordSmoke:
    """Smoke tests for tasks.text.last_word."""

    def test_importable(self):
        from tasks.text.last_word import last_word

        assert hasattr(last_word, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.text.last_word import last_word

        result = last_word.__wrapped_function__("hello world")
        assert result == "world"


# ---------------------------------------------------------------------------
# replace
# ---------------------------------------------------------------------------
class TestReplaceSmoke:
    """Smoke tests for tasks.text.replace."""

    def test_importable(self):
        from tasks.text.replace import replace

        assert hasattr(replace, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.text.replace import replace

        result = replace.__wrapped_function__("hello world", find="hello", replace_with="hi")
        assert result == "hi world"

    @pytest.mark.unit
    def test_regex_replace(self):
        from tasks.text.replace import replace

        result = replace.__wrapped_function__(
            "abc123def", find=r"\d+", replace_with="NUM", use_regex=True
        )
        assert result == "abcNUMdef"


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------
class TestSplitSmoke:
    """Smoke tests for tasks.text.split."""

    def test_importable(self):
        from tasks.text.split import split

        assert hasattr(split, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.text.split import split

        result = split.__wrapped_function__("a,b,c")
        assert result == ["a", "b", "c"]

    @pytest.mark.unit
    def test_custom_delimiter(self):
        from tasks.text.split import split

        result = split.__wrapped_function__("a|b|c", delimiter="|")
        assert result == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# split_into_chunks
# ---------------------------------------------------------------------------
class TestSplitIntoChunksSmoke:
    """Smoke tests for tasks.text.split_into_chunks."""

    def test_importable(self):
        from tasks.text.split_into_chunks import split_into_chunks

        assert hasattr(split_into_chunks, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.text.split_into_chunks import split_into_chunks

        result = split_into_chunks.__wrapped_function__("hello world")
        assert isinstance(result, list)
        assert len(result) >= 1

    @pytest.mark.unit
    def test_empty_text(self):
        from tasks.text.split_into_chunks import split_into_chunks

        result = split_into_chunks.__wrapped_function__("")
        assert result == []


# ---------------------------------------------------------------------------
# split_into_paragraphs
# ---------------------------------------------------------------------------
class TestSplitIntoParagraphsSmoke:
    """Smoke tests for tasks.text.split_into_paragraphs."""

    def test_importable(self):
        from tasks.text.split_into_paragraphs import split_into_paragraphs

        assert hasattr(split_into_paragraphs, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.text.split_into_paragraphs import split_into_paragraphs

        result = split_into_paragraphs.__wrapped_function__("hello\n\nworld")
        assert result == ["hello", "world"]


# ---------------------------------------------------------------------------
# split_into_sentences
# ---------------------------------------------------------------------------
class TestSplitIntoSentencesSmoke:
    """Smoke tests for tasks.text.split_into_sentences."""

    def test_importable(self):
        from tasks.text.split_into_sentences import split_into_sentences

        assert hasattr(split_into_sentences, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.text.split_into_sentences import split_into_sentences

        result = split_into_sentences.__wrapped_function__("Hello. World.")
        assert len(result) == 2
        assert result[0] == "Hello."


# ---------------------------------------------------------------------------
# split_into_words
# ---------------------------------------------------------------------------
class TestSplitIntoWordsSmoke:
    """Smoke tests for tasks.text.split_into_words."""

    def test_importable(self):
        from tasks.text.split_into_words import split_into_words

        assert hasattr(split_into_words, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.text.split_into_words import split_into_words

        result = split_into_words.__wrapped_function__("hello world")
        assert result == ["hello", "world"]


# ---------------------------------------------------------------------------
# split_lines
# ---------------------------------------------------------------------------
class TestSplitLinesSmoke:
    """Smoke tests for tasks.text.split_lines."""

    def test_importable(self):
        from tasks.text.split_lines import split_lines

        assert hasattr(split_lines, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.text.split_lines import split_lines

        result = split_lines.__wrapped_function__("hello\nworld")
        assert result == ["hello", "world"]


# ---------------------------------------------------------------------------
# substring
# ---------------------------------------------------------------------------
class TestSubstringSmoke:
    """Smoke tests for tasks.text.substring."""

    def test_importable(self):
        from tasks.text.substring import substring

        assert hasattr(substring, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.text.substring import substring

        result = substring.__wrapped_function__("hello world", start=0, end=5)
        assert result == "hello"

    @pytest.mark.unit
    def test_open_end(self):
        from tasks.text.substring import substring

        result = substring.__wrapped_function__("hello world", start=6)
        assert result == "world"


# ---------------------------------------------------------------------------
# trim
# ---------------------------------------------------------------------------
class TestTrimSmoke:
    """Smoke tests for tasks.text.trim."""

    def test_importable(self):
        from tasks.text.trim import trim

        assert hasattr(trim, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.text.trim import trim

        result = trim.__wrapped_function__("  hello  ")
        assert result == "hello"

    @pytest.mark.unit
    def test_lowercase(self):
        from tasks.text.trim import trim

        result = trim.__wrapped_function__("  HELLO  ", lowercase=True)
        assert result == "hello"


# ---------------------------------------------------------------------------
# truncate
# ---------------------------------------------------------------------------
class TestTruncateSmoke:
    """Smoke tests for tasks.text.truncate."""

    def test_importable(self):
        from tasks.text.truncate import truncate

        assert hasattr(truncate, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.text.truncate import truncate

        result = truncate.__wrapped_function__("hello world", max_length=5)
        assert len(result) == 5
        assert result == "he..."

    @pytest.mark.unit
    def test_no_truncation_needed(self):
        from tasks.text.truncate import truncate

        result = truncate.__wrapped_function__("hi", max_length=100)
        assert result == "hi"


# ---------------------------------------------------------------------------
# word_count
# ---------------------------------------------------------------------------
class TestWordCountSmoke:
    """Smoke tests for tasks.text.word_count."""

    def test_importable(self):
        from tasks.text.word_count import word_count

        assert hasattr(word_count, "__wrapped_function__")

    @pytest.mark.unit
    def test_basic_execution(self):
        from tasks.text.word_count import word_count

        result = word_count.__wrapped_function__("hello world")
        assert result == 2

    @pytest.mark.unit
    def test_empty_text(self):
        from tasks.text.word_count import word_count

        result = word_count.__wrapped_function__("")
        assert result == 0
