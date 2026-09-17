#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid ML pipeline for connective prediction.

Architecture:
  Stage 1 — MultiOutputClassifier (HistGradientBoosting) predicts sub-connectors: ¬, ∧, ∨
  Stage 2 — RandomForest predicts the main connective: →, ↔, ∧, ∨, ¬, ATOMIC

Features: 70+ handcrafted linguistic features + TF-IDF character n-grams (2-4).
Stage 1 predictions are stacked as additional features for Stage 2.
"""

import os
import re
import json
import unicodedata
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.feature_extraction import DictVectorizer

from logic_utils import standardize_formula_variables
from config import NEGATION_START_PHRASES, logger

class HybridPipelineTrainer:
    """Two-stage stacked ML pipeline for Spanish connective prediction.
    
    Trains sub-connector classifiers (¬/∧/∨ presence), then uses their
    predictions as features for a main-connector classifier (→/↔/∧/∨/¬/ATOMIC).
    """
    def __init__(self, data_path: str = 'test_hibrido.json'):
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.feature_columns: Optional[List[str]] = None

        self.clf_multi: Optional[MultiOutputClassifier] = None
        self.clf_main: Optional[RandomForestClassifier] = None
        self.boundary: Optional["BoundaryClassifier"] = None
        self.connective: Optional["ConnectiveSpanClassifier"] = None
        self.tagger = None

        self._load_data()
        self._extract_features()
        self._define_targets()

    def predict_with_confidence(self, sentence: str, allowed: Optional[set] = None) -> Tuple[str, float, Dict[str, bool]]:
        """Predict the main connective with confidence score and sub-connector flags."""
        feat = self.extract_advanced_features(sentence)
        hand_df = pd.DataFrame([feat])
        tfidf_df = self._tfidf_transform_single(sentence)
        X_user = pd.concat([hand_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
        X_user = X_user.reindex(columns=self.feature_columns, fill_value=0)

        pred_multi = self.clf_multi.predict(X_user)[0]
        X_user_aug = X_user.copy()
        X_user_aug['pred_neg'] = pred_multi[0]
        X_user_aug['pred_conj'] = pred_multi[1]
        X_user_aug['pred_disj'] = pred_multi[2]

        proba = self.clf_main.predict_proba(X_user_aug)[0]
        classes = list(self.clf_main.classes_)
        if allowed:
            masked = np.array(
                [p if cls_name in allowed else 0.0 for cls_name, p in zip(classes, proba)],
                dtype=float,
            )
            if float(masked.sum()) > 0:
                proba = masked
        confidence = float(proba.max())
        pred_main = classes[int(proba.argmax())]

        return pred_main, confidence, {
            'has_negation': bool(pred_multi[0]),
            'has_conjunction': bool(pred_multi[1]),
            'has_disjunction': bool(pred_multi[2])
        }

    def _load_data(self) -> None:
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.info(f"Warning: '{self.data_path}' no encontrado. Usando dataset mínimo.")
            data = [
                {"id": 1, "sentence": "No está lloviendo.", "propositions": {"P": "está lloviendo"}, "formula": "¬P", "connectives": ["¬"]},
                {"id": 2, "sentence": "El sistema no ha fallado.", "propositions": {"P": "el sistema ha fallado"}, "formula": "¬P", "connectives": ["¬"]},
                {"id": 3, "sentence": "Si llueve, me quedo en casa.", "propositions": {"P": "llueve", "Q": "me quedo en casa"}, "formula": "P → Q", "connectives": ["→"]},
                {"id": 4, "sentence": "Si estudias, apruebas el examen.", "propositions": {"P": "estudias", "Q": "apruebas el examen"}, "formula": "P → Q", "connectives": ["→"]},
                {"id": 5, "sentence": "Hace frío y está nublado.", "propositions": {"P": "hace frío", "Q": "está nublado"}, "formula": "P ∧ Q", "connectives": ["∧"]}
            ]
        self.df = pd.DataFrame(data)
        logger.info(f"Dataset cargado: {len(self.df)} oraciones.")

    def extract_advanced_features(self, sentence: str) -> Dict[str, float]:
        """Extract 80+ handcrafted linguistic features from a Spanish sentence.
        
        Groups: conjunction markers, disjunction markers, conditional markers,
        negation markers, biconditional markers, structural features, rare connectives.
        """
        features = {}
        s = sentence.lower().strip()
        len_s = len(s)

        # ── Conjunction markers ──
        features['freq_e_conj'] = len(re.findall(r'\be\b', s))
        features['has_sin_embargo'] = int('sin embargo' in s)
        features['has_no_obstante'] = int('no obstante' in s)
        features['has_a_pesar_de'] = int('a pesar de' in s)
        # ── Disjunction markers ──
        features['freq_u'] = len(re.findall(r'\bu\b', s))
        features['has_u_conj'] = int(bool(re.search(r'\bu\b', s)))
        features['double_o'] = int(bool(re.search(r'\bo\b.+\bo\b', s)))
        features['double_u'] = int(bool(re.search(r'\bu\b.+\bu\b', s)))
        features['has_o_ambas_cosas'] = int('o ambas cosas' in s or 'o ambos' in s)
        # ── Negation markers ──
        features['trailing_no'] = int(bool(re.search(r'\bno\s*$', s)))
        # ── Biconditional markers ──
        features['has_equivalente'] = int('equivalente a' in s and 'equivalente a decir' not in s)
        features['has_equivale'] = int('equivale a' in s)
        features['al_menos_with_u'] = int(bool(re.search(r'(al\s+menos|como\s+m[ií]nimo)\s+\w+\s+u\b', s)))
        features['al_menos_with_o'] = int(bool(re.search(r'(al\s+menos|como\s+m[ií]nimo)\s+\w+\s+o\b', s)))
        # ── Conditional markers ──
        features['freq_si'] = len(re.findall(r'\bsi\b', s))
        features['freq_entonces'] = len(re.findall(r'\bentonces\b', s))
        features['freq_si_y_solo_si'] = len(re.findall(r'si y solo si', s))
        features['freq_y'] = len(re.findall(r'\by\b', s))
        features['freq_o'] = len(re.findall(r'\bo\b', s))
        features['freq_no'] = len(re.findall(r'\bno\b', s))
        features['freq_comma'] = s.count(',')
        features['entonces_after_comma'] = int(re.search(r',\s*entonces', s) is not None)
        features['neg_before_verb'] = int(re.search(r'\bno\s+\w+', s) is not None)

        features['has_necesario_para'] = int(bool(re.search(r'\bnecesari[oa]s?\s+para\b', s)))
        features['has_suficiente_para'] = int(bool(re.search(r'\bsuficiente\s+para\b', s)))
        
        # ── Implication vs negation precedence ──
        strong_implication_markers = [r'\bnecesari[oa]s?\b', r'\bsuficiente\b', r'^si\b', r'\bentonces\b']
        first_impl_idx = len_s
        for pattern in strong_implication_markers:
            match = re.search(pattern, s)
            if match and match.start() < first_impl_idx:
                first_impl_idx = match.start()
                
        first_neg_idx = s.find('no ') if 'no ' in s else len_s
        
        features['implication_dominates_negation'] = int(first_impl_idx < first_neg_idx and first_impl_idx != len_s)

        features['aunque_with_negation'] = int(bool(re.search(r'\baunque\b.*\bno\b', s)))
        
        # ── Sentence-level structural features ──
        features['is_long_complex_sentence'] = int(len_s > 60 and features['freq_comma'] > 0)

        features['wrapper_conditional'] = int(s.startswith('si ') and re.search(r'\bentonces\b', s) is not None)

        features['double_conditional_and'] = int(bool(re.search(r'^si\s+.+?,\s*.+?,\s*y\s+si\s+', s)))

        words = re.findall(r'\b\w+\b', s)
        connectors = {'si', 'entonces', 'y', 'o', 'no', 'solo', 'sí', 'además'}
        content_words = [w for w in words if w not in connectors and len(w) > 2]
        features['num_atomic_hint'] = len(content_words)
        features['implicit_conditional'] = int(re.search(r'si\s+.+,\s*\w+', s) is not None)
        features['implicit_conditional_no_comma'] = int(s.startswith('si ') and ',' not in s)
        features['neg_before_conj'] = int(re.search(r'\bno\s+(y|o)\b', s) is not None)
        features['num_paren'] = s.count('(') + s.count(')')
        features['len_chars'] = len_s
        features['len_words'] = len(words)
        features['comma_ratio'] = features['freq_comma'] / (features['len_words'] + 1e-6)
        features['si_and_entonces'] = int(features['freq_si'] > 0 and features['freq_entonces'] > 0)
        features['neg_not_verb'] = int(re.search(r'\bno\s+(el|la|los|las|un|una)\b', s) is not None)
        features['bicond_phrase'] = int('si y solo si' in s)
        features['y_after_comma'] = len(re.findall(r',\s*y\b', s))
        features['o_after_comma'] = len(re.findall(r',\s*o\b', s))

        features['starts_with_negation_phrase'] = int(any(s.startswith(phrase) for phrase in NEGATION_START_PHRASES))

        negation_tokens = ['no', 'falso', 'es verdad']
        first_si = re.search(r'\bsi\b', s)
        if first_si:
            si_start = first_si.start()
            neg_before_si = any(re.search(r'\b' + re.escape(token) + r'\b', s[:si_start]) for token in negation_tokens)
            features['negation_before_conditional'] = int(neg_before_si)
        else:
            features['negation_before_conditional'] = 0

        first_neg = re.search(r'\b(no|falso)\b', s)
        if first_neg:
            neg_idx = first_neg.start()
            features['negation_distance_to_start'] = neg_idx / len_s if len_s > 0 else 1.0
        else:
            features['negation_distance_to_start'] = 1.0

        strong_markers = [r'\bentonces\b', r'\bo bien\b', r'\bsi y solo si\b']
        first_marker_start = None
        for pattern in strong_markers:
            match = re.search(pattern, s)
            if match:
                start = match.start()
                if first_marker_start is None or start < first_marker_start:
                    first_marker_start = start

        if first_neg and (first_marker_start is None or first_neg.start() < first_marker_start):
            features['negation_dominance'] = 1
        else:
            features['negation_dominance'] = 0

        features['starts_with_si'] = int(s.startswith('si '))
        features['starts_with_no_es'] = int(s.startswith('no es cierto') or s.startswith('es falso'))

        if 'si ' in s and ',' in s:
            antecedente = s.split(',')[0]
            features['disjunction_in_antecedent'] = int(' o ' in antecedente)
            features['conjunction_in_antecedent'] = int(' y ' in antecedente)
        else:
            features['disjunction_in_antecedent'] = 0
            features['conjunction_in_antecedent'] = 0

        features['inverted_si'] = int(' si ' in s and not s.startswith('si ') and 'si y solo si' not in s)
        features['freq_cuando'] = len(re.findall(r'\bcuando\b', s))
        features['has_a_menos_que'] = int('a menos que' in s)
        features['freq_pero'] = len(re.findall(r'\bpero\b', s))

        if ',' in s and ' si ' in s:
            consecuente = s.split(',', 1)[1] if ',' in s else ''
            features['nested_si_in_consequent'] = int(' si ' in consecuente and not consecuente.strip().startswith('si ') and 'si y solo si' not in consecuente)
        else:
            features['nested_si_in_consequent'] = 0

        # ── Rare/additional connectives ──
        features['freq_solo_si'] = int(
            bool(re.search(r'\bsolo\s+si\b', s)) and 'si y solo si' not in s)
        features['freq_siempre_que'] = int(
            bool(re.search(r'\bsiempre\s+(y\s+cuando|que)\b', s)))
        features['freq_salvo_excepto'] = int(
            bool(re.search(r'\b(salvo\s+que|excepto\s+si|a\s+no\s+ser\s+que)\b', s)))
        features['freq_basta'] = int(bool(re.search(r'\bbasta\b', s)))
        features['freq_suficiente'] = int(bool(re.search(r'\bsuficiente\b', s)))
        features['freq_necesario'] = int(bool(re.search(r'\bnecesari[oa]s?\b', s)))
        features['freq_ni'] = len(re.findall(r'\bni\b', s))
        features['ni_ni_pattern'] = int(bool(re.search(r'\bni\b.+\bni\b', s)))
        features['freq_aunque'] = int(bool(re.search(r'\baunque\b', s)))
        features['freq_o_bien'] = int(bool(re.search(r'\bo\s+bien\b', s)))
        features['freq_con_tal_que'] = int(bool(re.search(r'\bcon\s+tal\s+que\b', s)))
        features['freq_en_caso_de'] = int(bool(re.search(r'\ben\s+caso\s+de\b', s)))
        features['has_solo_si_not_bicond'] = int(
            'solo si' in s and 'si y solo si' not in s)

        features['freq_ademas_de'] = int(
            bool(re.search(r'\badem[aá]s\s+de\b', s)))
        features['freq_en_adicion'] = int(
            bool(re.search(r'\ben\s+adici[oó]n\s+a\b', s)))
        features['freq_al_menos'] = int(
            bool(re.search(r'\bal\s+menos\b|\bcomo\s+m[ií]nimo\b', s)))
        features['freq_nec_suf'] = int(
            bool(re.search(r'\bnecesari[oa]s?\s+y\s+suficiente\b', s)))
        features['freq_equivalente_decir'] = int(
            bool(re.search(r'\bequivalente\s+a\s+decir\b', s)))
        features['freq_viceversa'] = int(bool(re.search(r'\bviceversa\b', s)))
        features['freq_no_se_cumple'] = int(
            bool(re.search(r'\bno\s+se\s+cumple\b', s)))

        return features

    def _extract_features(self) -> None:
        hand_df = pd.DataFrame(self.df['sentence'].apply(self.extract_advanced_features).tolist())
        self._tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=300, sublinear_tf=True)
        tfidf_matrix = self._tfidf.fit_transform(self.df['sentence'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])

        self.X = pd.concat([hand_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
        self.feature_columns = self.X.columns.tolist()
        logger.info(f"Extraídas {hand_df.shape[1]} features artesanales + {tfidf_df.shape[1]} TF-IDF = {self.X.shape[1]} total.")

    def _tfidf_transform_single(self, sentence: str) -> pd.DataFrame:
        vec = self._tfidf.transform([sentence]).toarray()
        return pd.DataFrame(vec, columns=[f'tfidf_{i}' for i in range(vec.shape[1])])

    @staticmethod
    def _get_main_connective(formula: str) -> str:
        stripped = formula.strip()
        depth = 0
        for i, ch in enumerate(stripped):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif depth == 0 and ch in {'→', '↔', '∧', '∨'}:
                return ch
        if stripped.startswith('¬'):
            return '¬'
        return 'ATOMIC'

    def _define_targets(self) -> None:
        self.df['main_connective'] = self.df['formula'].apply(self._get_main_connective)
        self.y_main = self.df['main_connective']

        self.df['has_negation'] = self.df['formula'].apply(lambda f: int('¬' in f))
        self.df['has_conjunction'] = self.df['formula'].apply(lambda f: int('∧' in f))
        self.df['has_disjunction'] = self.df['formula'].apply(lambda f: int('∨' in f))
        self.y_multilabel = self.df[['has_negation', 'has_conjunction', 'has_disjunction']]

    def train_sub_connectors(self) -> None:
        """Train the multi-label sub-connector classifiers (¬/∧/∨ presence).
        
        Splits data, fits MultiOutputClassifier, and builds stacked features.
        """
        strat = self.y_main if self.y_main.value_counts().min() >= 2 else None
        try:
            X_train, X_test, y_multi_train, y_multi_test = train_test_split(
                self.X, self.y_multilabel, test_size=0.2, random_state=42, stratify=strat
            )
        except ValueError:
            X_train, X_test, y_multi_train, y_multi_test = train_test_split(
                self.X, self.y_multilabel, test_size=0.2, random_state=42
            )
        self.X_train = X_train
        self.X_test = X_test
        self.y_multi_train = y_multi_train
        self.y_multi_test = y_multi_test

        self.clf_multi = MultiOutputClassifier(
            HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=6, random_state=42)
        )
        self.clf_multi.fit(X_train, y_multi_train)
        y_multi_pred = self.clf_multi.predict(X_test)

        logger.info("\n--- Rendimiento del clasificador de sub-conectores ---")
        for i, label in enumerate(['has_negation', 'has_conjunction', 'has_disjunction']):
            acc = accuracy_score(y_multi_test.iloc[:, i], y_multi_pred[:, i])
            logger.info(f"{label} accuracy: {acc:.3f}")
        self._create_stacked_features()

    def _create_stacked_features(self) -> None:
        multi_preds_train = self.clf_multi.predict(self.X_train)
        multi_preds_test = self.clf_multi.predict(self.X_test)
        self.X_train_augmented = self.X_train.copy()
        self.X_test_augmented = self.X_test.copy()

        for i, label in enumerate(['pred_neg', 'pred_conj', 'pred_disj']):
            self.X_train_augmented[label] = multi_preds_train[:, i]
            self.X_test_augmented[label] = multi_preds_test[:, i]

    def train_main_connector(self) -> None:
        """Train the main-connector classifier (→/↔/∧/∨/¬/ATOMIC) using augmented features."""
        self.y_main_train = self.y_main[self.X_train.index]
        self.y_main_test = self.y_main[self.X_test.index]

        self.clf_main = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        self.clf_main.fit(self.X_train_augmented, self.y_main_train)
        y_main_pred = self.clf_main.predict(self.X_test_augmented)

        logger.info("\n--- Rendimiento del clasificador de conector principal (Holdout) ---")
        logger.info(f"Accuracy: {accuracy_score(self.y_main_test, y_main_pred):.3f}")
        logger.info(classification_report(self.y_main_test, y_main_pred, zero_division=0))
        # Deployed models: refit on the full training set after reporting holdout.
        self.clf_multi.fit(self.X, self.y_multilabel)
        X_full = self.X.copy()
        full_multi = self.clf_multi.predict(self.X)
        for i, label in enumerate(['pred_neg', 'pred_conj', 'pred_disj']):
            X_full[label] = full_multi[:, i]
        self.clf_main.fit(X_full, self.y_main)

    def cross_validate_main(self, n_splits: int = 5) -> None:
        """Run stratified k-fold cross-validation on the main connector classifier."""
        X_full_augmented = self.X.copy()
        full_multi_preds = self.clf_multi.predict(self.X)
        for i, label in enumerate(['pred_neg', 'pred_conj', 'pred_disj']):
            X_full_augmented[label] = full_multi_preds[:, i]

        logger.info("\n--- Validación Cruzada Estratificada ---")
        try:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.clf_main, X_full_augmented, self.y_main, cv=skf, scoring='f1_macro')
            logger.info(f"F1-Macro Promedio ({n_splits}-Fold): {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        except ValueError as e:
            min_class_size = self.y_main.value_counts().min()
            new_splits = max(2, min(n_splits - 1, min_class_size))
            logger.info(f"No hay suficientes muestras para {n_splits}-Fold (error: {e}). Reintentando con {new_splits} pliegues...")
            if new_splits >= 2:
                skf = StratifiedKFold(n_splits=new_splits, shuffle=True, random_state=42)
                cv_scores = cross_val_score(self.clf_main, X_full_augmented, self.y_main, cv=skf, scoring='f1_macro')
                logger.info(f"F1-Macro Promedio ({new_splits}-Fold): {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    def predict(self, sentence: str) -> Tuple[str, Dict[str, bool]]:
        """Predict main connective and sub-connector flags for a single sentence."""
        feat = self.extract_advanced_features(sentence)
        hand_df = pd.DataFrame([feat])
        
        tfidf_df = self._tfidf_transform_single(sentence)
        
        X_user = pd.concat([hand_df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
        X_user = X_user.reindex(columns=self.feature_columns, fill_value=0)

        pred_multi = self.clf_multi.predict(X_user)[0]
        
        X_user_aug = X_user.copy()
        X_user_aug['pred_neg'] = pred_multi[0]
        X_user_aug['pred_conj'] = pred_multi[1]
        X_user_aug['pred_disj'] = pred_multi[2]

        pred_main = self.clf_main.predict(X_user_aug)[0]
        
        return pred_main, {
            'has_negation': bool(pred_multi[0]),
            'has_conjunction': bool(pred_multi[1]),
            'has_disjunction': bool(pred_multi[2])
        }

    def save_models(self, filepath: str = 'models.joblib') -> None:
        """Serialize classifiers, TF-IDF vectorizer, and feature columns to disk."""
        try:
            os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
            data = {
                'clf_multi': self.clf_multi,
                'clf_main': self.clf_main,
                'tfidf': self._tfidf,
                'feature_columns': self.feature_columns,
                'boundary': None if self.boundary is None else self.boundary.to_dict(),
                'connective': None if self.connective is None else self.connective.to_dict(),
                'tagger': None if self.tagger is None else self.tagger.to_dict(),
            }
            joblib.dump(data, filepath)
            logger.info(f"Modelos guardados exitosamente en '{filepath}'.")
        except Exception as e:
            logger.error(f"Error al guardar los modelos: {e}")

    def predict_connective(self, sentence : str) -> Tuple[str,Dict[str,bool]]:
        main_pred, flags = self.predict(sentence)
        return main_pred, flags

    def load_models(self, filepath: str = 'models.joblib') -> bool:
        """Load serialized models from disk. Returns True on success, False if file is missing."""
        if not os.path.exists(filepath):
            return False
        try:
            data = joblib.load(filepath)
            self.clf_multi = data['clf_multi']
            self.clf_main = data['clf_main']
            self._tfidf = data['tfidf']
            self.feature_columns = data['feature_columns']
            if data.get('boundary'):
                self.boundary = BoundaryClassifier.from_dict(data['boundary'])
            if data.get('connective'):
                self.connective = ConnectiveSpanClassifier.from_dict(data['connective'])
            if data.get('tagger'):
                from span_tagger import JointSpanTagger
                self.tagger = JointSpanTagger.from_dict(data['tagger'])
            logger.info(f"Modelos cargados en memoria desde '{filepath}'.")
            return True
        except Exception as e:
            logger.error(f"Error al cargar los modelos: {e}")
            return False

    def train_boundary_classifier(self) -> None:
        from span_tagger import JointSpanTagger, gold_cues

        items = self.df.to_dict(orient="records")
        self.tagger = JointSpanTagger()
        self.tagger.train_from_dataset(items)
        self.boundary = BoundaryClassifier()
        self.boundary.train_from_dataset(items, cue_fn=gold_cues)

    def predict_boundary_proba(self, sentence: str, cue) -> float:
        if self.boundary is None:
            from pipeline_parser import gold_is_boundary
            return 1.0 if (cue.force or gold_is_boundary(sentence, cue, None)) else 0.25
        return self.boundary.predict_boundary_proba(sentence, cue)

    def predict_cues(self, sentence: str):
        if self.tagger is not None and getattr(self.tagger, "_ready", False):
            return self.tagger.predict_cues(sentence)
        return []


class BoundaryClassifier:
    """Stage-2 clause-boundary model with batch HGB + online SGD."""

    def __init__(self, feedback_path: str = "data/feedback.jsonl", refit_every: int = 20):
        self.batch_clf = HistGradientBoostingClassifier(
            max_iter=120, learning_rate=0.08, max_depth=5, random_state=42
        )
        self.sgd = SGDClassifier(loss="log_loss", random_state=42)
        self.feedback_path = feedback_path
        self.refit_every = refit_every
        self.gold_X: Optional[np.ndarray] = None
        self.gold_y: Optional[np.ndarray] = None
        self.feedback_X: List[List[float]] = []
        self.feedback_y: List[int] = []
        self._sgd_ready = False
        self._batch_ready = False
        self._feedback_events = 0

    def train_from_dataset(self, items: List[dict], cue_fn=None) -> None:
        from pipeline_parser import extract_boundary_features, gold_is_boundary, vectorize_features
        from span_tagger import gold_cues

        get_cues = cue_fn or gold_cues
        X, y = [], []
        for item in items:
            sentence = item.get("sentence") or ""
            for cue in get_cues(sentence, item):
                X.append(vectorize_features(extract_boundary_features(sentence, cue)))
                y.append(int(gold_is_boundary(sentence, cue, item)))
        self.gold_X = np.asarray(X, dtype=float)
        self.gold_y = np.asarray(y, dtype=int)
        if self.gold_X.size == 0:
            logger.info("No hay cues para entrenar fronteras.")
            return
        unique = set(self.gold_y.tolist())
        if len(unique) >= 2:
            self.batch_clf.fit(self.gold_X, self.gold_y)
            self._batch_ready = True
            self.sgd.partial_fit(self.gold_X, self.gold_y, classes=np.array([0, 1]))
            self._sgd_ready = True
            pred = self.batch_clf.predict(self.gold_X)
            logger.info(
                f"Fronteras: n={len(y)} pos={int(self.gold_y.sum())} "
                f"acc={accuracy_score(self.gold_y, pred):.3f} "
                f"f1={f1_score(self.gold_y, pred, zero_division=0):.3f}"
            )
        else:
            logger.info(f"Fronteras: una sola clase ({unique}); se usarán heurísticas.")

    def predict_boundary_proba(self, sentence: str, cue) -> float:
        from pipeline_parser import extract_boundary_features, gold_is_boundary, vectorize_features

        if cue.force:
            return 0.99
        vec = np.asarray([vectorize_features(extract_boundary_features(sentence, cue))], dtype=float)
        p_batch = None
        p_sgd = None
        if self._batch_ready:
            p_batch = float(self.batch_clf.predict_proba(vec)[0, 1])
        if self._sgd_ready:
            try:
                p_sgd = float(self.sgd.predict_proba(vec)[0, 1])
            except Exception:
                p_sgd = p_batch
        from pipeline_parser import is_np_coordination, looks_verbal
        left, right = sentence[:cue.start], sentence[cue.end:]
        prior = None
        if cue.kind in {"AND", "OR", "SINO"}:
            if looks_verbal(left) and looks_verbal(right):
                prior = 0.62
            elif is_np_coordination(left, right):
                prior = 0.6
        if p_batch is None and p_sgd is None:
            base = 0.9 if gold_is_boundary(sentence, cue, None) else 0.2
            return max(base, prior or 0.0)
        if p_sgd is None:
            p = p_batch
        elif p_batch is None:
            p = p_sgd
        else:
            w = min(0.5, 0.04 * self._feedback_events)
            p = (1.0 - w) * p_batch + w * p_sgd
        if prior is not None:
            p = max(p, prior)
        return p

    def learn_labels(self, sentence: str, labels: List[dict], meta: Optional[dict] = None) -> None:
        from pipeline_parser import Cue, detect_cues, extract_boundary_features, vectorize_features

        cues = detect_cues(sentence)
        X, y = [], []
        for lab in labels:
            cue = next((c for c in cues if c.start == lab["start"] and c.end == lab["end"]), None)
            if cue is None:
                cue = Cue(lab["start"], lab["end"], lab.get("kind", "AND"), "∧", lab.get("text", ""), False)
            X.append(vectorize_features(extract_boundary_features(sentence, cue)))
            y.append(int(lab["boundary"]))
        if not X:
            return
        arr_x, arr_y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        classes = np.array([0, 1])
        self.sgd.partial_fit(arr_x, arr_y, classes=classes)
        self._sgd_ready = True
        self.feedback_X.extend(X)
        self.feedback_y.extend(y)
        self._feedback_events += 1
        self._append_feedback(sentence, labels, meta)
        if self._feedback_events % self.refit_every == 0:
            self.refit()

    def refit(self) -> None:
        if self.gold_X is None or self.gold_X.size == 0:
            return
        extra_x = np.asarray(self.feedback_X, dtype=float) if self.feedback_X else None
        extra_y = np.asarray(self.feedback_y, dtype=int) if self.feedback_y else None
        if extra_x is not None and extra_x.size:
            X = np.vstack([self.gold_X, extra_x])
            y = np.concatenate([self.gold_y, extra_y])
        else:
            X, y = self.gold_X, self.gold_y
        if len(set(y.tolist())) < 2:
            return
        self.batch_clf.fit(X, y)
        self._batch_ready = True
        self.sgd.partial_fit(X, y, classes=np.array([0, 1]))
        logger.info(f"Refit de fronteras con {len(y)} ejemplos (feedback={len(self.feedback_y)}).")

    def _append_feedback(self, sentence: str, labels: List[dict], meta: Optional[dict]) -> None:
        os.makedirs(os.path.dirname(self.feedback_path) or ".", exist_ok=True)
        row = {"sentence": sentence, "labels": labels, "meta": meta or {}}
        with open(self.feedback_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def to_dict(self) -> dict:
        return {
            "batch_clf": self.batch_clf if self._batch_ready else None,
            "sgd": self.sgd if self._sgd_ready else None,
            "gold_X": self.gold_X,
            "gold_y": self.gold_y,
            "feedback_X": self.feedback_X,
            "feedback_y": self.feedback_y,
            "feedback_path": self.feedback_path,
            "refit_every": self.refit_every,
            "feedback_events": self._feedback_events,
            "batch_ready": self._batch_ready,
            "sgd_ready": self._sgd_ready,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BoundaryClassifier":
        obj = cls(feedback_path=data.get("feedback_path", "data/feedback.jsonl"),
                  refit_every=int(data.get("refit_every", 20)))
        if data.get("batch_clf") is not None:
            obj.batch_clf = data["batch_clf"]
            obj._batch_ready = bool(data.get("batch_ready", True))
        if data.get("sgd") is not None:
            obj.sgd = data["sgd"]
            obj._sgd_ready = bool(data.get("sgd_ready", True))
        obj.gold_X = data.get("gold_X")
        obj.gold_y = data.get("gold_y")
        obj.feedback_X = list(data.get("feedback_X") or [])
        obj.feedback_y = list(data.get("feedback_y") or [])
        obj._feedback_events = int(data.get("feedback_events") or 0)
        return obj


class ConnectiveSpanClassifier:
    """Token BIO tagger: each word is O or part of a connective.

    Regex cues and the text between gold propositions are weak labels only.
    Function-word dropout teaches paraphrases ('si solo si' ≈ 'si y solo si')
    without adding lexicon entries. Inference is argmax over tokens, not a parser.
    """

    _FUNC = {"y", "e", "o", "u", "de", "a", "que", "el", "la"}
    _CONTINUE = {"IFF", "ONLY_IF", "NEG_BLOCK", "XOR_TAIL", "NO_SOLO", "SINO"}

    def __init__(self):
        self.vec: Optional[DictVectorizer] = None
        self.clf: Optional[LogisticRegression] = None
        self._ready = False

    def train_from_dataset(self, items: List[dict]) -> None:
        feats: List[dict] = []
        labels: List[str] = []
        for tokens, labs in self._iter_sequences(items):
            for i, lab in enumerate(labs):
                feats.append(self._token_features(tokens, i))
                labels.append(lab)
        if len(set(labels)) < 2:
            logger.info("Connective-BIO: clases insuficientes.")
            return
        self.vec = DictVectorizer(sparse=True)
        X = self.vec.fit_transform(feats)
        self.clf = LogisticRegression(
            max_iter=400, C=0.4, class_weight="balanced", solver="saga", random_state=42,
        )
        self.clf.fit(X, labels)
        self._ready = True
        pred = self.clf.predict(X)
        logger.info(
            f"Connective-BIO: n={len(labels)} acc={accuracy_score(labels, pred):.3f} "
            f"clases={sorted(set(labels))}"
        )

    def _iter_sequences(self, items: List[dict]):
        from pipeline_parser import detect_cues

        for item in items:
            sentence = item.get("sentence") or ""
            tokens = list(re.finditer(r"\S+", sentence))
            if not tokens:
                continue
            gold = self._gold_spans(sentence, item, detect_cues(sentence))
            labs = self._spans_to_bio(tokens, gold)
            yield tokens, labs
            for t2, l2 in self._dropout_sequences(tokens, labs):
                yield t2, l2
            for t2, l2 in self._mask_interior_sequences(tokens, labs):
                yield t2, l2

    def _gold_spans(self, sentence: str, item: dict, cues) -> List[Tuple[int, int, str]]:
        spans = [(c.start, c.end, c.kind) for c in cues]
        spans = self._merge_if_only_if(spans)
        glue = self._glue_span(sentence, item.get("propositions") or {})
        kind = self._glue_kind_from_formula(item.get("formula") or "", glue, sentence)
        if glue and kind:
            a, b = glue
            if 0 < (b - a) <= 56:
                spans.append((a, b, kind))
        return spans

    @staticmethod
    def _merge_if_only_if(spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
        """Weak label: adjacent IF + ONLY_IF is IFF (si [y] solo si, any filler)."""
        ordered = sorted(spans, key=lambda s: (s[0], s[1]))
        used = [False] * len(ordered)
        out = []
        for i, (a, b, k) in enumerate(ordered):
            if used[i]:
                continue
            merged = False
            if k == "IF":
                for j in range(i + 1, len(ordered)):
                    a2, b2, k2 = ordered[j]
                    if used[j]:
                        continue
                    if a2 > b + 8:
                        break
                    if k2 == "ONLY_IF" and a2 <= b + 8:
                        out.append((a, b2, "IFF"))
                        used[i] = used[j] = True
                        merged = True
                        break
            if not merged:
                out.append((a, b, k))
                used[i] = True
        return out

    def _spans_to_bio(self, tokens, spans: List[Tuple[int, int, str]]) -> List[str]:
        from pipeline_parser import PREC

        labs = ["O"] * len(tokens)
        # Low precedence first so IFF/ONLY_IF overwrite nested IF/AND.
        for start, end, kind in sorted(spans, key=lambda s: (PREC.get(s[2], 0), s[1] - s[0])):
            hit = [i for i, tok in enumerate(tokens) if not (tok.end() <= start or tok.start() >= end)]
            for k, i in enumerate(hit):
                labs[i] = f"B-{kind}" if k == 0 else f"I-{kind}"
        return labs

    def _dropout_sequences(self, tokens, labs):
        droppable = []
        for i, (tok, lab) in enumerate(zip(tokens, labs)):
            if lab == "O":
                continue
            kind = lab.split("-", 1)[-1]
            if kind not in self._CONTINUE:
                continue
            if self._norm(tok.group()) in self._FUNC:
                droppable.append(i)
        if not droppable:
            return
        keep = [i for i in range(len(tokens)) if i not in set(droppable)]
        if len(keep) >= 2:
            yield [tokens[i] for i in keep], [labs[i] for i in keep]
        for d in droppable:
            keep_one = [i for i in range(len(tokens)) if i != d]
            yield [tokens[i] for i in keep_one], [labs[i] for i in keep_one]

    def _mask_interior_sequences(self, tokens, labs):
        """Hide interior IFF wording so the si … si frame transfers to unseen fillers."""
        idxs = [
            i for i, lab in enumerate(labs)
            if lab.endswith("-IFF") and self._norm(tokens[i].group()) not in {"si", "y", "e"}
        ]
        if not idxs:
            return
        yield self._retoken(tokens, idxs, "xxxx"), list(labs)

    @staticmethod
    def _retoken(tokens, idxs, filler: str):
        class _T:
            def __init__(self, match, text):
                self._m = match
                self._t = text
            def group(self, *args):
                return self._t
            def start(self):
                return self._m.start()
            def end(self):
                return self._m.end()
        out = []
        for i, tok in enumerate(tokens):
            out.append(_T(tok, filler) if i in set(idxs) else tok)
        return out

    def _token_features(self, tokens, i: int) -> dict:
        n = len(tokens)
        w = self._norm(tokens[i].group())
        prev = self._norm(tokens[i - 1].group()) if i else "<s>"
        nxt = self._norm(tokens[i + 1].group()) if i + 1 < n else "</s>"
        prev2 = self._norm(tokens[i - 2].group()) if i >= 2 else "<s>"
        nxt2 = self._norm(tokens[i + 2].group()) if i + 2 < n else "</s>"
        feat = {
            "w": w,
            "p": prev,
            "n": nxt,
            "p2": prev2,
            "n2": nxt2,
            "wp": f"{prev}|{w}",
            "wn": f"{w}|{nxt}",
            "tri": f"{prev}|{w}|{nxt}",
            "suf2": w[-2:],
            "suf3": w[-3:],
            "pre3": w[:3],
            "len": str(min(len(w), 15)),
            "prev_si": str(prev == "si"),
            "next_si": str(nxt == "si"),
            "flank_si": str(prev == "si" and nxt == "si"),
            "si_win": str((prev == "si" or prev2 == "si") and (nxt == "si" or nxt2 == "si")),
        }
        padded = f" {w} "
        for k in range(2, 5):
            for j in range(len(padded) - k + 1):
                feat[f"c{k}_{padded[j:j + k]}"] = True
        blob = f"{prev} {w} {nxt}"
        for k in range(3, 6):
            for j in range(0, max(len(blob) - k + 1, 0)):
                feat[f"t{k}_{blob[j:j + k]}"] = True
        return feat

    @staticmethod
    def _norm(text: str) -> str:
        folded = unicodedata.normalize("NFD", text.lower())
        folded = "".join(c for c in folded if unicodedata.category(c) != "Mn")
        folded = re.sub(r"[^\wñ]+", "", folded)
        return folded

    @staticmethod
    def _glue_span(sentence: str, props: dict) -> Optional[Tuple[int, int]]:
        found = []
        low = sentence.lower()
        for phrase in props.values():
            p = str(phrase).lower().strip()
            if len(p) < 3:
                continue
            idx = low.find(p)
            if idx >= 0:
                found.append((idx, idx + len(p)))
        found.sort()
        if len(found) < 2:
            return None
        a, b = found[0][1], found[1][0]
        if b <= a or (b - a) > 56:
            return None
        return a, b

    @staticmethod
    def _glue_kind_from_formula(formula: str, glue: Optional[Tuple[int, int]], sentence: str) -> Optional[str]:
        stripped = (formula or "").strip()
        depth = 0
        main = "ATOMIC"
        for ch in stripped:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif depth == 0 and ch in {"→", "↔", "∧", "∨"}:
                main = ch
                break
        else:
            if stripped.startswith("¬"):
                main = "¬"
        g = ""
        if glue:
            g = sentence[glue[0]:glue[1]].lower()
        # Glue is noisy for AND/IF (it can swallow y/o inside an implication).
        # Only use it for phrase-level connectives whose wording varies.
        if main == "↔":
            return "IFF"
        if main == "→" and re.search(r"\bsolo\s+si\b", g) and "si y solo" not in g:
            return "ONLY_IF"
        if main == "¬" and len(g) >= 8:
            return "NEG_BLOCK"
        return None

    def predict_spans(self, sentence: str):
        from pipeline_parser import Cue, KIND_TO_OP, ML_FORCE_KINDS, nms_cues

        if not self._ready or self.clf is None or self.vec is None:
            return []
        tokens = list(re.finditer(r"\S+", sentence))
        if not tokens:
            return []
        X = self.vec.transform([self._token_features(tokens, i) for i in range(len(tokens))])
        proba = self.clf.predict_proba(X)
        classes = list(self.clf.classes_)
        labels = []
        confs = []
        for dist in proba:
            top = int(np.argmax(dist))
            labels.append(classes[top])
            confs.append(float(dist[top]))
        cues = self._bio_to_cues(sentence, tokens, labels, confs)
        return nms_cues(cues)

    def _bio_to_cues(self, sentence: str, tokens, labels: List[str], confs: List[float]):
        from pipeline_parser import Cue, KIND_TO_OP, ML_FORCE_KINDS

        cues = []
        i = 0
        n = len(tokens)
        while i < n:
            lab = labels[i]
            if lab == "O" or "-" not in lab:
                i += 1
                continue
            prefix, kind = lab.split("-", 1)
            if prefix == "I":
                i += 1
                continue
            start = tokens[i].start()
            end = tokens[i].end()
            ps = [confs[i]]
            i += 1
            if kind in self._CONTINUE:
                while i < n:
                    lab2 = labels[i]
                    if lab2 in {f"I-{kind}", f"B-{kind}"} or lab2 == kind:
                        if lab2.startswith("B-") and tokens[i].start() > end + 2:
                            break
                        if lab2.startswith("B-") and kind not in self._CONTINUE:
                            break
                        end = tokens[i].end()
                        ps.append(confs[i])
                        i += 1
                        continue
                    break
            p = float(np.mean(ps))
            n_tok = len(sentence[start:end].split())
            min_p = 0.38 if n_tok >= 2 else 0.48
            if p < min_p:
                continue
            op = KIND_TO_OP.get(kind, "∧")
            text = sentence[start:end]
            force = kind in ML_FORCE_KINDS
            cues.append(Cue(start, end, kind, op, text, force, prob=p, is_boundary=True))
        return cues

    def to_dict(self) -> dict:
        return {"vec": self.vec, "clf": self.clf, "ready": self._ready}

    @classmethod
    def from_dict(cls, data: dict) -> "ConnectiveSpanClassifier":
        obj = cls()
        obj.vec = data.get("vec")
        obj.clf = data.get("clf")
        obj._ready = bool(data.get("ready") and obj.clf is not None and obj.vec is not None)
        if obj.vec is None and data.get("pipe") is not None:
            obj._ready = False
        return obj

