"""
Contract test suite for src_constrain_synthesizer component.
Tests parse_synthesis_output and write_artifacts functions.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from constrain.synthesizer import parse_synthesis_output, write_artifacts


class TestParseSynthesisOutput:
    """Test suite for parse_synthesis_output function."""
    
    def test_parse_synthesis_output_happy_path_basic(self):
        """Parse valid LLM output with both delimiters present."""
        raw = "--- PROMPT ---\nThis is the prompt content\n--- CONSTRAINTS ---\nthese are constraints"
        
        result = parse_synthesis_output(raw)
        
        # Verify result is a tuple with 2 elements
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        # Verify extracted content
        assert result[0] == "This is the prompt content"
        assert result[1] == "these are constraints"
    
    def test_parse_synthesis_output_happy_path_multiline(self):
        """Parse output with multiline content in both sections."""
        raw = "--- PROMPT ---\nLine 1\nLine 2\nLine 3\n--- CONSTRAINTS ---\nConstraint 1\nConstraint 2"
        
        result = parse_synthesis_output(raw)
        
        # Verify multiline strings are properly extracted
        assert result[0] == "Line 1\nLine 2\nLine 3"
        assert result[1] == "Constraint 1\nConstraint 2"
    
    def test_parse_synthesis_output_edge_case_extra_whitespace(self):
        """Parse output with extra whitespace around content."""
        raw = "--- PROMPT ---\n\n  Prompt with spaces  \n\n--- CONSTRAINTS ---\n\n  Constraints with spaces  \n\n"
        
        result = parse_synthesis_output(raw)
        
        # Verify content is stripped of leading/trailing whitespace
        assert result[0] == "Prompt with spaces"
        assert result[1] == "Constraints with spaces"
    
    def test_parse_synthesis_output_edge_case_empty_content(self):
        """Parse output where sections have no content between delimiters."""
        raw = "--- PROMPT ---\n--- CONSTRAINTS ---"
        
        result = parse_synthesis_output(raw)
        
        # Verify empty strings are returned
        assert result[0] == ""
        assert result[1] == ""
    
    def test_parse_synthesis_output_edge_case_special_characters(self):
        """Parse output with special characters and unicode in content."""
        raw = "--- PROMPT ---\nSpecial: @#$%^&* 日本語 émojis 🎉\n--- CONSTRAINTS ---\nYAML: key: \"value\""
        
        result = parse_synthesis_output(raw)
        
        # Verify special characters and unicode are preserved
        assert "@#$%^&*" in result[0]
        assert "日本語" in result[0]
        assert "🎉" in result[0]
        assert 'key: "value"' in result[1]
    
    def test_parse_synthesis_output_edge_case_delimiters_in_content(self):
        """Parse output where delimiter text appears within content sections.

        The parser uses str.find() which locates the *first* occurrence of
        '--- CONSTRAINTS ---'.  When that marker appears inside the PROMPT
        section, the parser splits there, so the prompt is truncated and
        the constraints section includes the trailing content plus the real
        constraints block.
        """
        raw = "--- PROMPT ---\nContent mentioning --- CONSTRAINTS --- marker\n--- CONSTRAINTS ---\nSome constraints"

        result = parse_synthesis_output(raw)

        # find() picks the first '--- CONSTRAINTS ---' (inside the content),
        # so prompt is only the text before that first occurrence.
        assert result[0] == "Content mentioning"
        # Everything after the first '--- CONSTRAINTS ---' delimiter becomes
        # the constraints section, including the remainder of the original
        # prompt line and the real constraints block.
        assert result[1] == "marker\n--- CONSTRAINTS ---\nSome constraints"
    
    def test_parse_synthesis_output_error_missing_prompt_delimiter(self):
        """Raise error when PROMPT delimiter is missing."""
        raw = "No prompt delimiter\n--- CONSTRAINTS ---\nConstraints here"
        
        # Verify exception is raised
        with pytest.raises(Exception) as exc_info:
            parse_synthesis_output(raw)
        
        # Verify error message mentions missing delimiters
        assert "missing" in str(exc_info.value).lower() or "delimiter" in str(exc_info.value).lower()
    
    def test_parse_synthesis_output_error_missing_constraints_delimiter(self):
        """Raise error when CONSTRAINTS delimiter is missing."""
        raw = "--- PROMPT ---\nPrompt here\nNo constraints delimiter"
        
        # Verify exception is raised
        with pytest.raises(Exception) as exc_info:
            parse_synthesis_output(raw)
        
        # Verify error message mentions missing delimiters
        assert "missing" in str(exc_info.value).lower() or "delimiter" in str(exc_info.value).lower()
    
    def test_parse_synthesis_output_error_missing_both_delimiters(self):
        """Raise error when both delimiters are missing."""
        raw = "Just some random text with no delimiters at all"
        
        # Verify exception is raised
        with pytest.raises(Exception) as exc_info:
            parse_synthesis_output(raw)
        
        # Verify error message mentions missing delimiters
        assert "missing" in str(exc_info.value).lower() or "delimiter" in str(exc_info.value).lower()
    
    def test_parse_synthesis_output_invariant_marker_constants(self):
        """Verify parser uses exact marker strings as constants."""
        raw = "--- PROMPT ---\nContent\n--- CONSTRAINTS ---\nMore"
        
        result = parse_synthesis_output(raw)
        
        # Verify parsing succeeds with exact marker strings
        assert result is not None
        assert len(result) == 2
        assert result[0] == "Content"
        assert result[1] == "More"


class TestWriteArtifacts:
    """Test suite for write_artifacts function."""
    
    def test_write_artifacts_happy_path_basic(self, tmp_path):
        """Write prompt.md and constraints.yaml to a directory."""
        prompt_md = "# Prompt Title\nPrompt content"
        constraints_yaml = "version: 1.0\nconstraints: []"
        output_dir = tmp_path / "test_output"
        
        result = write_artifacts(prompt_md, constraints_yaml, output_dir, overwrite=False)
        
        # Verify output_dir exists
        assert output_dir.exists()
        assert output_dir.is_dir()
        
        # Verify files exist
        prompt_path = output_dir / "prompt.md"
        constraints_path = output_dir / "constraints.yaml"
        assert prompt_path.exists()
        assert constraints_path.exists()
        
        # Verify content
        assert prompt_path.read_text(encoding="utf-8") == prompt_md
        assert constraints_path.read_text(encoding="utf-8") == constraints_yaml
        
        # Verify returned paths are Path objects
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], Path)
        assert isinstance(result[1], Path)
        
        # Verify returned paths point to created files
        assert result[0] == prompt_path
        assert result[1] == constraints_path
    
    def test_write_artifacts_happy_path_with_path_object(self, tmp_path):
        """Write artifacts using Path object for output_dir."""
        prompt_md = "Path test prompt"
        constraints_yaml = "path: test"
        output_dir = tmp_path / "path_test"
        
        # Pass Path object instead of string
        result = write_artifacts(prompt_md, constraints_yaml, output_dir, overwrite=False)
        
        # Verify files are created successfully
        assert (output_dir / "prompt.md").exists()
        assert (output_dir / "constraints.yaml").exists()
        
        # Verify content matches input
        assert (output_dir / "prompt.md").read_text(encoding="utf-8") == prompt_md
        assert (output_dir / "constraints.yaml").read_text(encoding="utf-8") == constraints_yaml
    
    def test_write_artifacts_happy_path_with_overwrite(self, tmp_path):
        """Overwrite existing files when overwrite=True."""
        output_dir = tmp_path / "overwrite_test"
        output_dir.mkdir()
        
        # Create existing files with old content
        old_prompt = "Old prompt content"
        old_constraints = "old: constraints"
        (output_dir / "prompt.md").write_text(old_prompt, encoding="utf-8")
        (output_dir / "constraints.yaml").write_text(old_constraints, encoding="utf-8")
        
        # Write new content with overwrite=True
        new_prompt = "New prompt content"
        new_constraints = "new: constraints"
        result = write_artifacts(new_prompt, new_constraints, output_dir, overwrite=True)
        
        # Verify old content is replaced
        prompt_content = (output_dir / "prompt.md").read_text(encoding="utf-8")
        constraints_content = (output_dir / "constraints.yaml").read_text(encoding="utf-8")
        
        assert prompt_content != old_prompt
        assert constraints_content != old_constraints
        
        # Verify new content is present
        assert prompt_content == new_prompt
        assert constraints_content == new_constraints
    
    def test_write_artifacts_edge_case_creates_directory(self, tmp_path):
        """Create output directory if it doesn't exist."""
        output_dir = tmp_path / "nonexistent_dir"
        
        # Ensure directory doesn't exist before call
        assert not output_dir.exists()
        
        result = write_artifacts("Test", "test", output_dir, overwrite=False)
        
        # Verify directory is created
        assert output_dir.exists()
        assert output_dir.is_dir()
        
        # Verify files exist in new directory
        assert (output_dir / "prompt.md").exists()
        assert (output_dir / "constraints.yaml").exists()
    
    def test_write_artifacts_edge_case_empty_content(self, tmp_path):
        """Write files with empty content strings."""
        output_dir = tmp_path / "empty_test"
        
        result = write_artifacts("", "", output_dir, overwrite=False)
        
        # Verify files exist
        prompt_path = output_dir / "prompt.md"
        constraints_path = output_dir / "constraints.yaml"
        assert prompt_path.exists()
        assert constraints_path.exists()
        
        # Verify reading files returns empty strings
        assert prompt_path.read_text(encoding="utf-8") == ""
        assert constraints_path.read_text(encoding="utf-8") == ""
    
    def test_write_artifacts_edge_case_unicode_content(self, tmp_path):
        """Write files with unicode content in UTF-8 encoding."""
        output_dir = tmp_path / "unicode_test"
        prompt_md = "Unicode: 日本語 émojis 🎉 中文"
        constraints_yaml = "unicode_key: '中文值'"
        
        result = write_artifacts(prompt_md, constraints_yaml, output_dir, overwrite=False)
        
        # Verify files contain unicode characters
        prompt_content = (output_dir / "prompt.md").read_text(encoding="utf-8")
        constraints_content = (output_dir / "constraints.yaml").read_text(encoding="utf-8")
        
        assert "日本語" in prompt_content
        assert "🎉" in prompt_content
        assert "中文" in prompt_content
        assert "中文值" in constraints_content
        
        # Verify content can be read back correctly
        assert prompt_content == prompt_md
        assert constraints_content == constraints_yaml
    
    def test_write_artifacts_edge_case_large_content(self, tmp_path):
        """Write files with large content strings."""
        output_dir = tmp_path / "large_test"
        
        # Generate large strings (10KB+)
        large_prompt = "A" * 15000
        large_yaml = "B" * 12000
        
        result = write_artifacts(large_prompt, large_yaml, output_dir, overwrite=False)
        
        # Verify files are created
        prompt_path = output_dir / "prompt.md"
        constraints_path = output_dir / "constraints.yaml"
        assert prompt_path.exists()
        assert constraints_path.exists()
        
        # Verify file size matches content length
        assert prompt_path.stat().st_size >= 15000
        assert constraints_path.stat().st_size >= 12000
        
        # Verify content can be read back
        assert prompt_path.read_text(encoding="utf-8") == large_prompt
        assert constraints_path.read_text(encoding="utf-8") == large_yaml
    
    def test_write_artifacts_error_file_exists_no_overwrite(self, tmp_path):
        """Raise error when files exist and overwrite=False."""
        output_dir = tmp_path / "exists_test"
        output_dir.mkdir()
        
        # Create existing files with original content
        original_prompt = "Original prompt"
        original_constraints = "original: constraints"
        (output_dir / "prompt.md").write_text(original_prompt, encoding="utf-8")
        (output_dir / "constraints.yaml").write_text(original_constraints, encoding="utf-8")
        
        # Try to write with overwrite=False
        with pytest.raises(Exception) as exc_info:
            write_artifacts("New content", "new constraints", output_dir, overwrite=False)
        
        # Verify exception mentions file_exists
        error_msg = str(exc_info.value).lower()
        assert "file" in error_msg or "exist" in error_msg
        
        # Verify original files are not modified
        assert (output_dir / "prompt.md").read_text(encoding="utf-8") == original_prompt
        assert (output_dir / "constraints.yaml").read_text(encoding="utf-8") == original_constraints
    
    def test_write_artifacts_invariant_fixed_filenames(self, tmp_path):
        """Verify output filenames are always 'prompt.md' and 'constraints.yaml'."""
        output_dir = tmp_path / "filename_test"
        
        result = write_artifacts("Test", "test", output_dir, overwrite=False)
        
        # Verify returned paths end with correct filenames
        assert result[0].name == "prompt.md"
        assert result[1].name == "constraints.yaml"
        
        # Verify no other files are created
        files = list(output_dir.iterdir())
        assert len(files) == 2
        filenames = {f.name for f in files}
        assert filenames == {"prompt.md", "constraints.yaml"}
    
    def test_write_artifacts_invariant_utf8_encoding(self, tmp_path):
        """Verify files are always written in UTF-8 encoding."""
        output_dir = tmp_path / "encoding_test"
        prompt_md = "UTF-8 test: café"
        constraints_yaml = "encoding: test"
        
        result = write_artifacts(prompt_md, constraints_yaml, output_dir, overwrite=False)
        
        # Verify files can be read with UTF-8 encoding
        prompt_content = (output_dir / "prompt.md").read_text(encoding="utf-8")
        constraints_content = (output_dir / "constraints.yaml").read_text(encoding="utf-8")
        
        # Verify special characters are preserved
        assert "café" in prompt_content
        assert prompt_content == prompt_md
        assert constraints_content == constraints_yaml


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_integration_parse_and_write_cycle(self, tmp_path):
        """Integration test: parse LLM output then write artifacts."""
        raw = "--- PROMPT ---\n# Task\nGenerate code\n--- CONSTRAINTS ---\nversion: 1.0\nmax_lines: 100"
        
        # Parse the output
        prompt_content, constraints_content = parse_synthesis_output(raw)
        
        # Verify parsing extracts correct content
        assert "# Task" in prompt_content
        assert "Generate code" in prompt_content
        assert "version: 1.0" in constraints_content
        assert "max_lines: 100" in constraints_content
        
        # Write artifacts
        output_dir = tmp_path / "integration_test"
        prompt_path, constraints_path = write_artifacts(
            prompt_content, constraints_content, output_dir, overwrite=False
        )
        
        # Verify writing creates files with parsed content
        assert prompt_path.exists()
        assert constraints_path.exists()
        
        # Verify files can be read back and match parsed values
        read_prompt = prompt_path.read_text(encoding="utf-8")
        read_constraints = constraints_path.read_text(encoding="utf-8")
        
        assert read_prompt == prompt_content
        assert read_constraints == constraints_content
        assert "# Task" in read_prompt
        assert "version: 1.0" in read_constraints
