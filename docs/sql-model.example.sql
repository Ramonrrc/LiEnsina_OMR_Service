CREATE TABLE evaluation_corrections (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  evaluation_id uuid NOT NULL,
  class_id uuid NOT NULL,
  student_id uuid NOT NULL,
  status text NOT NULL,
  image_url text,
  suggested_score numeric(5,2) NOT NULL DEFAULT 0,
  final_score numeric(5,2),
  correct_count integer NOT NULL DEFAULT 0,
  wrong_count integer NOT NULL DEFAULT 0,
  blank_count integer NOT NULL DEFAULT 0,
  multiple_count integer NOT NULL DEFAULT 0,
  total_questions integer NOT NULL DEFAULT 0,
  confidence numeric(5,4) NOT NULL DEFAULT 0,
  requires_review boolean NOT NULL DEFAULT true,
  should_retake_image boolean NOT NULL DEFAULT false,
  failures jsonb NOT NULL DEFAULT '[]'::jsonb,
  detected_answers jsonb NOT NULL DEFAULT '[]'::jsonb,
  answer_key jsonb NOT NULL DEFAULT '[]'::jsonb,
  reviewed_by_id uuid,
  reviewed_at timestamptz,
  created_by_id uuid NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (evaluation_id, student_id)
);

CREATE INDEX evaluation_corrections_evaluation_id_idx ON evaluation_corrections (evaluation_id);
CREATE INDEX evaluation_corrections_class_id_idx ON evaluation_corrections (class_id);
CREATE INDEX evaluation_corrections_student_id_idx ON evaluation_corrections (student_id);
