# -*- coding: utf-8 -*-
"""
LLM-based model for generating candidate points.
"""

import json
import re
import os
import numpy as np
from google import genai


class LLMOptimizer:
    """
    Optimizer that uses a Large Language Model to suggest candidate points.
    """

    def __init__(self, api_key, model_family="models/gemini-1.5-flash"):
        """
        Initialize the LLM optimizer.

        Args:
            api_key: API key for the generative AI model
            model_family: Model family to use for generation (e.g., "gemini-1.5-flash")
        """
        self.client = genai.Client(api_key=api_key)
        self.model_family = model_family

    def get_models_with_generate_content_action(self, model_filter):
        """
        Returns a list of models that support the generateContent action.

        Args:
            model_filter: Filter for model names

        Returns:
            List of model names
        """
        exact_matches = []
        partial_matches = []
        for m in self.client.models.list():
            for action in m.supported_actions:
                if action == "generateContent":
                    if m.name == model_filter:
                        # Exact match
                        exact_matches.append(m.name)
                    elif model_filter in m.name:
                        # Partial match
                        partial_matches.append(m.name)

        # Return exact matches first, then partial matches
        return exact_matches + partial_matches

    def validate_candidates(self, candidates, dimension_names):
        """
        Validate that the generated candidates have the expected structure.

        Args:
            candidates: List of candidate dictionaries
            dimension_names: List of expected dimension names

        Returns:
            bool: True if valid, False otherwise
        """
        if not isinstance(candidates, list) or len(candidates) == 0:
            print("Error: Candidates must be a non-empty list")
            return False

        for candidate in candidates:
            # Check that all dimensions are present
            for dim in dimension_names:
                if dim not in candidate:
                    print(f"Error: Missing dimension {dim} in candidate")
                    return False

                # Check that dimension values are numbers
                if not isinstance(candidate[dim], (int, float)):
                    print(f"Error: Value for dimension {dim} is not a number")
                    return False

            # Check that value key is present and is a number
            if "value" not in candidate:
                print("Error: Missing 'value' key in candidate")
                return False

            if not isinstance(candidate["value"], (int, float)):
                print("Error: 'value' is not a number")
                return False

        return True

    def generate_json_with_retry(self, prompt, model_name, dimension_names,
                                 max_retries=3):
        """
        Generate JSON content with retry logic and validation.

        Args:
            prompt: Text prompt for the model
            model_name: Name of the model to use
            dimension_names: List of expected dimension names for validation
            max_retries: Maximum number of retry attempts

        Returns:
            List of validated candidate dictionaries or None if failed
        """
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt+1} with model {model_name}")

                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )

                # Check if response is valid
                if response is None:
                    raise RuntimeError("Model failed to generate content")

                # Extract the JSON array from the response text
                json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
                if not json_match:
                    raise ValueError("Failed to extract JSON array from the response")

                json_string = json_match.group(0)

                # Parse the JSON
                candidates = json.loads(json_string)

                # Validate the structure of the candidates
                if self.validate_candidates(candidates, dimension_names):
                    return candidates
                else:
                    raise ValueError("Generated candidates have invalid structure")

            except Exception as e:
                print(f"Attempt {attempt+1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    print(f"Max retries ({max_retries}) reached. Could not generate valid candidates.")
                    return None

        return None

    def generate_new_candidates(self, data_points, function_values,
                                bounds=None, n_candidates=1, max_retries=3):
        """
        Generate new candidate points for evaluating a blackbox function using the Gemini API.

        Args:
            data_points: Array of shape (n_points, d) with evaluated input points
            function_values: Array of function evaluations for each data point
            bounds: List of d tuples [(min1, max1), (min2, max2), ...] defining the search domain
            n_candidates: Number of candidate points to generate
            max_retries: Maximum number of retry attempts per model

        Returns:
            Array of shape (n_candidates, d) containing the new candidate points
            Array of shape (n_candidates,) containing predicted function values
        """
        # Set default bounds to a 3D unit cube if not provided
        if bounds is None:
            d = 3
            bounds = [(0, 1)] * d
        else:
            d = len(bounds)

        data_points = np.array(data_points)
        function_values = np.array(function_values)

        # Generate dimension names as x1, x2, ..., xd
        dimension_names = [f"x{i+1}" for i in range(d)]

        # Construct the prompt
        prompt = self._create_optimization_prompt(
            data_points, function_values, dimension_names, bounds, n_candidates
        )

        # Get list of available models
        list_of_models = self.get_models_with_generate_content_action(self.model_family)
        candidate_data = None

        # Try each model in turn until we get valid candidates
        for model_name in list_of_models:
            candidate_data = self.generate_json_with_retry(
                prompt=prompt,
                model_name=model_name,
                dimension_names=dimension_names,
                max_retries=max_retries
            )

            if candidate_data is not None:
                print(f"Successfully generated valid candidates with model {model_name}")
                break

        if candidate_data is None:
            raise RuntimeError("All models failed to generate valid candidates after multiple retries")

        # Assemble candidate points along with their predicted values
        candidates = []
        predicted_f = []

        for candidate in candidate_data:
            # Get the coordinate values for each dimension
            candidate_point = [candidate[dim] for dim in dimension_names]
            candidates.append(candidate_point)
            predicted_f.append(candidate["value"])

        return np.array(candidates), np.array(predicted_f)

    def _create_optimization_prompt(self, data_points, function_values, dimension_names, bounds, n_candidates):
        """
        Create a prompt for generating candidate points.

        Args:
            data_points: Array of shape (n_points, d) with evaluated input points
            function_values: Array of function evaluations for each data point
            dimension_names: List of dimension names
            bounds: List of d tuples [(min1, max1), (min2, max2), ...] defining the search domain
            n_candidates: Number of candidate points to generate

        Returns:
            Prompt string
        """
        # Construct the prompt header with context
        prompt = f"Suggest {n_candidates} new candidate point(s) for maximizing a blackbox function in a {len(dimension_names)}-dimensional search space.\n\n"
        prompt += "Below are some examples of previously evaluated points with their corresponding function values:\n"

        # Build the in-context examples as a JSON list
        examples = []
        for point, value in zip(data_points, function_values):
            example = {dimension_names[i]: round(float(point[i]), 3) for i in range(len(dimension_names))}
            example["value"] = round(float(value), 3)
            examples.append(example)
        prompt += json.dumps(examples, indent=4) + "\n\n"

        # Append bounding box details for each dimension
        prompt += "The search space is defined by the following bounding boxes:\n"
        for i, (low, high) in enumerate(bounds):
            prompt += f"   {dimension_names[i]}_min: {low:.3f}, {dimension_names[i]}_max: {high:.3f}\n"
        prompt += "\n"

        # Additional instructions
        prompt += (
            "Based on the examples above, suggest candidate points that balance exploration (sampling new regions) "
            "with exploitation (focusing on promising areas where function values are good). Each candidate point "
            "must lie within the specified bounding boxes. In addition, predict an estimated function value for each candidate.\n\n"
            "Return the suggestions in the following JSON format exactly, without any additional text:\n"
        )

        # Construct the JSON schema dynamically, including the predicted "value"
        schema_fields = ', '.join([f'"{dim}": float' for dim in dimension_names] + ['"value": float'])
        prompt += f"[{{{schema_fields}}}]\n"

        return prompt

